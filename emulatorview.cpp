#include "emulatorview.h"

#include <QMetaObject>
#include <QVector2D>
#include <algorithm>
#include <cstring>

namespace {

constexpr auto VertexShader = R"(
attribute vec2 position;
attribute vec2 texCoord;
varying vec2 vTexCoord;

void main() {
    vTexCoord = texCoord;
    gl_Position = vec4(position, 0.0, 1.0);
}
)";

constexpr auto FragmentShader = R"(
uniform sampler2D frameTexture;
uniform vec2 textureSize;
uniform vec2 outputSize;
uniform int scalingFilter;
uniform float sharpAmount;
uniform float edgeBlend;
uniform float nearestHold;
varying vec2 vTexCoord;

float luma(vec4 c) {
    return dot(c.rgb, vec3(0.299, 0.587, 0.114));
}

float colorDistance(vec4 a, vec4 b) {
    vec3 pa = a.rgb * a.a;
    vec3 pb = b.rgb * b.a;
    vec3 d = pa - pb;

    float y = dot(d, vec3(0.299, 0.587, 0.114));
    vec2 chroma = vec2(d.r - d.g, d.b - d.g);
    float da = a.a - b.a;

    return y * y * 2.5 + dot(chroma, chroma) * 0.5 + da * da;
}

vec4 sampleAt(vec2 pixel, vec2 offset) {
    return texture2D(frameTexture, (pixel + offset + vec2(0.5)) / textureSize);
}

vec4 cubicWeights(float t) {
    float t2 = t * t;
    float t3 = t2 * t;

    return vec4(
        -0.5 * t + t2 - 0.5 * t3,
         1.0 - 2.5 * t2 + 1.5 * t3,
         0.5 * t + 2.0 * t2 - 1.5 * t3,
        -0.5 * t2 + 0.5 * t3
    );
}

vec4 cubicRow(vec2 pixel, float y, vec4 wx) {
    return sampleAt(pixel, vec2(-1.0, y)) * wx.x +
           sampleAt(pixel, vec2( 0.0, y)) * wx.y +
           sampleAt(pixel, vec2( 1.0, y)) * wx.z +
           sampleAt(pixel, vec2( 2.0, y)) * wx.w;
}

vec4 catmullRomAntiRing(vec2 pixel, vec2 local) {
    vec4 wx = cubicWeights(local.x);
    vec4 wy = cubicWeights(local.y);

    vec4 r0 = cubicRow(pixel, -1.0, wx);
    vec4 r1 = cubicRow(pixel,  0.0, wx);
    vec4 r2 = cubicRow(pixel,  1.0, wx);
    vec4 r3 = cubicRow(pixel,  2.0, wx);

    vec4 c = r0 * wy.x + r1 * wy.y + r2 * wy.z + r3 * wy.w;

    vec4 c00 = sampleAt(pixel, vec2(0.0, 0.0));
    vec4 c10 = sampleAt(pixel, vec2(1.0, 0.0));
    vec4 c01 = sampleAt(pixel, vec2(0.0, 1.0));
    vec4 c11 = sampleAt(pixel, vec2(1.0, 1.0));

    vec4 mn = min(min(c00, c10), min(c01, c11));
    vec4 mx = max(max(c00, c10), max(c01, c11));

    return clamp(c, mn, mx);
}

vec4 refinedScale() {
    vec2 pixel_position = (vTexCoord * textureSize) - vec2(0.5);
    vec2 pixel = floor(pixel_position);
    vec2 local = fract(pixel_position);

    vec4 center     = sampleAt(pixel, vec2( 0.0,  0.0));
    vec4 left       = sampleAt(pixel, vec2(-1.0,  0.0));
    vec4 right      = sampleAt(pixel, vec2( 1.0,  0.0));
    vec4 up         = sampleAt(pixel, vec2( 0.0, -1.0));
    vec4 down       = sampleAt(pixel, vec2( 0.0,  1.0));
    vec4 up_left    = sampleAt(pixel, vec2(-1.0, -1.0));
    vec4 up_right   = sampleAt(pixel, vec2( 1.0, -1.0));
    vec4 down_left  = sampleAt(pixel, vec2(-1.0,  1.0));
    vec4 down_right = sampleAt(pixel, vec2( 1.0,  1.0));

    vec4 base = catmullRomAntiRing(pixel, local);

    float gx =
        luma(right) - luma(left) +
        0.5 * ((luma(up_right) - luma(up_left)) +
               (luma(down_right) - luma(down_left)));

    float gy =
        luma(down) - luma(up) +
        0.5 * ((luma(down_left) - luma(up_left)) +
               (luma(down_right) - luma(up_right)));

    float ax = abs(gx);
    float ay = abs(gy);
    float edge = smoothstep(0.04, 0.25, sqrt(gx * gx + gy * gy));

    vec4 noCross = base;

    if (ax > ay * 1.20) {
        float xSide = step(0.5, local.x);
        vec4 a = sampleAt(pixel, vec2(xSide, 0.0));
        vec4 b = sampleAt(pixel, vec2(xSide, 1.0));
        noCross = mix(a, b, local.y);
    } else if (ay > ax * 1.20) {
        float ySide = step(0.5, local.y);
        vec4 a = sampleAt(pixel, vec2(0.0, ySide));
        vec4 b = sampleAt(pixel, vec2(1.0, ySide));
        noCross = mix(a, b, local.x);
    }

    vec4 outColor = mix(base, noCross, edge * edgeBlend);

    vec2 nearestOffset = step(vec2(0.5), local);
    vec4 nearestColor = sampleAt(pixel, nearestOffset);

    vec2 nearestDist = abs(local - nearestOffset);
    float nearestWeight =
        1.0 - smoothstep(0.12, 0.48, max(nearestDist.x, nearestDist.y));

    outColor = mix(outColor, nearestColor, nearestWeight * edge * nearestHold);

    vec4 lowpass = (center * 4.0 + left + right + up + down) * 0.125;
    vec4 detail = outColor - lowpass;

    float contrast = max(
        max(colorDistance(center, left), colorDistance(center, right)),
        max(colorDistance(center, up),   colorDistance(center, down))
    );

    float sharpenMask = smoothstep(0.001, 0.06, contrast);
    outColor += detail * sharpAmount * sharpenMask;

    vec4 mn = min(center, min(min(left, right), min(up, down)));
    mn = min(mn, min(min(up_left, up_right), min(down_left, down_right)));

    vec4 mx = max(center, max(max(left, right), max(up, down)));
    mx = max(mx, max(max(up_left, up_right), max(down_left, down_right)));

    return clamp(outColor, mn, mx);
}

float xbrzDistYCbCr(vec3 pixA, vec3 pixB) {
    const vec3 w = vec3(0.2627, 0.6780, 0.0593);
    const float scaleB = 0.5 / (1.0 - w.b);
    const float scaleR = 0.5 / (1.0 - w.r);
    vec3 diff = pixA - pixB;
    float y = dot(diff.rgb, w);
    float cb = scaleB * (diff.b - y);
    float cr = scaleR * (diff.r - y);

    return sqrt(y * y + cb * cb + cr * cr);
}

bool xbrzEq(vec3 pixA, vec3 pixB) {
    return all(equal(pixA, pixB));
}

bool xbrzNeq(vec3 pixA, vec3 pixB) {
    return !xbrzEq(pixA, pixB);
}

bool xbrzIsPixEqual(vec3 pixA, vec3 pixB) {
    return xbrzDistYCbCr(pixA, pixB) < 30.0 / 255.0;
}

float xbrzLeftRatio(vec2 center, vec2 origin, vec2 direction, vec2 scale) {
    vec2 p0 = center - origin;
    vec2 projection = direction * (dot(p0, direction) / dot(direction, direction));
    vec2 distance_vector = p0 - projection;
    vec2 orthogonal = vec2(-direction.y, direction.x);
    float side = sign(dot(p0, orthogonal));
    float distance = side * length(distance_vector * scale);

    return smoothstep(-sqrt(2.0) / 2.0, sqrt(2.0) / 2.0, distance);
}

vec3 xbrzSample(vec2 coord, vec2 offset) {
    return texture2D(frameTexture, coord + offset / textureSize).rgb;
}

vec4 xbrzFreescale() {
    const int BLEND_NONE = 0;
    const int BLEND_NORMAL = 1;
    const int BLEND_DOMINANT = 2;
    const float STEEP_DIRECTION_THRESHOLD = 2.2;
    const float DOMINANT_DIRECTION_THRESHOLD = 3.6;

    vec2 scale = outputSize / textureSize;
    vec2 pos = fract(vTexCoord * textureSize) - vec2(0.5, 0.5);
    vec2 coord = vTexCoord - pos / textureSize;

    vec3 A = xbrzSample(coord, vec2(-1.0, -1.0));
    vec3 B = xbrzSample(coord, vec2( 0.0, -1.0));
    vec3 C = xbrzSample(coord, vec2( 1.0, -1.0));
    vec3 D = xbrzSample(coord, vec2(-1.0,  0.0));
    vec3 E = xbrzSample(coord, vec2( 0.0,  0.0));
    vec3 F = xbrzSample(coord, vec2( 1.0,  0.0));
    vec3 G = xbrzSample(coord, vec2(-1.0,  1.0));
    vec3 H = xbrzSample(coord, vec2( 0.0,  1.0));
    vec3 I = xbrzSample(coord, vec2( 1.0,  1.0));

    ivec4 blendResult = ivec4(BLEND_NONE, BLEND_NONE, BLEND_NONE, BLEND_NONE);

    if (!((xbrzEq(E, F) && xbrzEq(H, I)) || (xbrzEq(E, H) && xbrzEq(F, I)))) {
        float dist_H_F = xbrzDistYCbCr(G, E) + xbrzDistYCbCr(E, C) + xbrzDistYCbCr(xbrzSample(coord, vec2(0.0, 2.0)), I) + xbrzDistYCbCr(I, xbrzSample(coord, vec2(2.0, 0.0))) + 4.0 * xbrzDistYCbCr(H, F);
        float dist_E_I = xbrzDistYCbCr(D, H) + xbrzDistYCbCr(H, xbrzSample(coord, vec2(1.0, 2.0))) + xbrzDistYCbCr(B, F) + xbrzDistYCbCr(F, xbrzSample(coord, vec2(2.0, 1.0))) + 4.0 * xbrzDistYCbCr(E, I);
        bool dominantGradient = DOMINANT_DIRECTION_THRESHOLD * dist_H_F < dist_E_I;
        blendResult.z = ((dist_H_F < dist_E_I) && xbrzNeq(E, F) && xbrzNeq(E, H)) ? (dominantGradient ? BLEND_DOMINANT : BLEND_NORMAL) : BLEND_NONE;
    }

    if (!((xbrzEq(D, E) && xbrzEq(G, H)) || (xbrzEq(D, G) && xbrzEq(E, H)))) {
        float dist_G_E = xbrzDistYCbCr(xbrzSample(coord, vec2(-2.0, 1.0)), D) + xbrzDistYCbCr(D, B) + xbrzDistYCbCr(xbrzSample(coord, vec2(-1.0, 2.0)), H) + xbrzDistYCbCr(H, F) + 4.0 * xbrzDistYCbCr(G, E);
        float dist_D_H = xbrzDistYCbCr(xbrzSample(coord, vec2(-2.0, 0.0)), G) + xbrzDistYCbCr(G, xbrzSample(coord, vec2(0.0, 2.0))) + xbrzDistYCbCr(A, E) + xbrzDistYCbCr(E, I) + 4.0 * xbrzDistYCbCr(D, H);
        bool dominantGradient = DOMINANT_DIRECTION_THRESHOLD * dist_D_H < dist_G_E;
        blendResult.w = ((dist_G_E > dist_D_H) && xbrzNeq(E, D) && xbrzNeq(E, H)) ? (dominantGradient ? BLEND_DOMINANT : BLEND_NORMAL) : BLEND_NONE;
    }

    if (!((xbrzEq(B, C) && xbrzEq(E, F)) || (xbrzEq(B, E) && xbrzEq(C, F)))) {
        float dist_E_C = xbrzDistYCbCr(D, B) + xbrzDistYCbCr(B, xbrzSample(coord, vec2(1.0, -2.0))) + xbrzDistYCbCr(H, F) + xbrzDistYCbCr(F, xbrzSample(coord, vec2(2.0, -1.0))) + 4.0 * xbrzDistYCbCr(E, C);
        float dist_B_F = xbrzDistYCbCr(A, E) + xbrzDistYCbCr(E, I) + xbrzDistYCbCr(xbrzSample(coord, vec2(0.0, -2.0)), C) + xbrzDistYCbCr(C, xbrzSample(coord, vec2(2.0, 0.0))) + 4.0 * xbrzDistYCbCr(B, F);
        bool dominantGradient = DOMINANT_DIRECTION_THRESHOLD * dist_B_F < dist_E_C;
        blendResult.y = ((dist_E_C > dist_B_F) && xbrzNeq(E, B) && xbrzNeq(E, F)) ? (dominantGradient ? BLEND_DOMINANT : BLEND_NORMAL) : BLEND_NONE;
    }

    if (!((xbrzEq(A, B) && xbrzEq(D, E)) || (xbrzEq(A, D) && xbrzEq(B, E)))) {
        float dist_D_B = xbrzDistYCbCr(xbrzSample(coord, vec2(-2.0, 0.0)), A) + xbrzDistYCbCr(A, xbrzSample(coord, vec2(0.0, -2.0))) + xbrzDistYCbCr(G, E) + xbrzDistYCbCr(E, C) + 4.0 * xbrzDistYCbCr(D, B);
        float dist_A_E = xbrzDistYCbCr(xbrzSample(coord, vec2(-2.0, -1.0)), D) + xbrzDistYCbCr(D, H) + xbrzDistYCbCr(xbrzSample(coord, vec2(-1.0, -2.0)), B) + xbrzDistYCbCr(B, F) + 4.0 * xbrzDistYCbCr(A, E);
        bool dominantGradient = DOMINANT_DIRECTION_THRESHOLD * dist_D_B < dist_A_E;
        blendResult.x = ((dist_D_B < dist_A_E) && xbrzNeq(E, D) && xbrzNeq(E, B)) ? (dominantGradient ? BLEND_DOMINANT : BLEND_NORMAL) : BLEND_NONE;
    }

    vec3 res = E;

    if (blendResult.z != BLEND_NONE) {
        float dist_F_G = xbrzDistYCbCr(F, G);
        float dist_H_C = xbrzDistYCbCr(H, C);
        bool doLineBlend = blendResult.z == BLEND_DOMINANT || !((blendResult.y != BLEND_NONE && !xbrzIsPixEqual(E, G)) || (blendResult.w != BLEND_NONE && !xbrzIsPixEqual(E, C)) || (xbrzIsPixEqual(G, H) && xbrzIsPixEqual(H, I) && xbrzIsPixEqual(I, F) && xbrzIsPixEqual(F, C) && !xbrzIsPixEqual(E, I)));
        vec2 origin = vec2(0.0, 1.0 / sqrt(2.0));
        vec2 direction = vec2(1.0, -1.0);
        if (doLineBlend) {
            bool haveShallowLine = STEEP_DIRECTION_THRESHOLD * dist_F_G <= dist_H_C && xbrzNeq(E, G) && xbrzNeq(D, G);
            bool haveSteepLine = STEEP_DIRECTION_THRESHOLD * dist_H_C <= dist_F_G && xbrzNeq(E, C) && xbrzNeq(B, C);
            origin = haveShallowLine ? vec2(0.0, 0.25) : vec2(0.0, 0.5);
            direction.x += haveShallowLine ? 1.0 : 0.0;
            direction.y -= haveSteepLine ? 1.0 : 0.0;
        }
        vec3 blendPix = mix(H, F, step(xbrzDistYCbCr(E, F), xbrzDistYCbCr(E, H)));
        res = mix(res, blendPix, xbrzLeftRatio(pos, origin, direction, scale));
    }

    if (blendResult.w != BLEND_NONE) {
        float dist_H_A = xbrzDistYCbCr(H, A);
        float dist_D_I = xbrzDistYCbCr(D, I);
        bool doLineBlend = blendResult.w == BLEND_DOMINANT || !((blendResult.z != BLEND_NONE && !xbrzIsPixEqual(E, A)) || (blendResult.x != BLEND_NONE && !xbrzIsPixEqual(E, I)) || (xbrzIsPixEqual(A, D) && xbrzIsPixEqual(D, G) && xbrzIsPixEqual(G, H) && xbrzIsPixEqual(H, I) && !xbrzIsPixEqual(E, G)));
        vec2 origin = vec2(-1.0 / sqrt(2.0), 0.0);
        vec2 direction = vec2(1.0, 1.0);
        if (doLineBlend) {
            bool haveShallowLine = STEEP_DIRECTION_THRESHOLD * dist_H_A <= dist_D_I && xbrzNeq(E, A) && xbrzNeq(B, A);
            bool haveSteepLine = STEEP_DIRECTION_THRESHOLD * dist_D_I <= dist_H_A && xbrzNeq(E, I) && xbrzNeq(F, I);
            origin = haveShallowLine ? vec2(-0.25, 0.0) : vec2(-0.5, 0.0);
            direction.y += haveShallowLine ? 1.0 : 0.0;
            direction.x += haveSteepLine ? 1.0 : 0.0;
        }
        vec3 blendPix = mix(H, D, step(xbrzDistYCbCr(E, D), xbrzDistYCbCr(E, H)));
        res = mix(res, blendPix, xbrzLeftRatio(pos, origin, direction, scale));
    }

    if (blendResult.y != BLEND_NONE) {
        float dist_B_I = xbrzDistYCbCr(B, I);
        float dist_F_A = xbrzDistYCbCr(F, A);
        bool doLineBlend = blendResult.y == BLEND_DOMINANT || !((blendResult.x != BLEND_NONE && !xbrzIsPixEqual(E, I)) || (blendResult.z != BLEND_NONE && !xbrzIsPixEqual(E, A)) || (xbrzIsPixEqual(I, F) && xbrzIsPixEqual(F, C) && xbrzIsPixEqual(C, B) && xbrzIsPixEqual(B, A) && !xbrzIsPixEqual(E, C)));
        vec2 origin = vec2(1.0 / sqrt(2.0), 0.0);
        vec2 direction = vec2(-1.0, -1.0);
        if (doLineBlend) {
            bool haveShallowLine = STEEP_DIRECTION_THRESHOLD * dist_B_I <= dist_F_A && xbrzNeq(E, I) && xbrzNeq(H, I);
            bool haveSteepLine = STEEP_DIRECTION_THRESHOLD * dist_F_A <= dist_B_I && xbrzNeq(E, A) && xbrzNeq(D, A);
            origin = haveShallowLine ? vec2(0.25, 0.0) : vec2(0.5, 0.0);
            direction.y -= haveShallowLine ? 1.0 : 0.0;
            direction.x -= haveSteepLine ? 1.0 : 0.0;
        }
        vec3 blendPix = mix(F, B, step(xbrzDistYCbCr(E, B), xbrzDistYCbCr(E, F)));
        res = mix(res, blendPix, xbrzLeftRatio(pos, origin, direction, scale));
    }

    if (blendResult.x != BLEND_NONE) {
        float dist_D_C = xbrzDistYCbCr(D, C);
        float dist_B_G = xbrzDistYCbCr(B, G);
        bool doLineBlend = blendResult.x == BLEND_DOMINANT || !((blendResult.w != BLEND_NONE && !xbrzIsPixEqual(E, C)) || (blendResult.y != BLEND_NONE && !xbrzIsPixEqual(E, G)) || (xbrzIsPixEqual(C, B) && xbrzIsPixEqual(B, A) && xbrzIsPixEqual(A, D) && xbrzIsPixEqual(D, G) && !xbrzIsPixEqual(E, A)));
        vec2 origin = vec2(0.0, -1.0 / sqrt(2.0));
        vec2 direction = vec2(-1.0, 1.0);
        if (doLineBlend) {
            bool haveShallowLine = STEEP_DIRECTION_THRESHOLD * dist_D_C <= dist_B_G && xbrzNeq(E, C) && xbrzNeq(F, C);
            bool haveSteepLine = STEEP_DIRECTION_THRESHOLD * dist_B_G <= dist_D_C && xbrzNeq(E, G) && xbrzNeq(H, G);
            origin = haveShallowLine ? vec2(0.0, -0.25) : vec2(0.0, -0.5);
            direction.x -= haveShallowLine ? 1.0 : 0.0;
            direction.y += haveSteepLine ? 1.0 : 0.0;
        }
        vec3 blendPix = mix(D, B, step(xbrzDistYCbCr(E, B), xbrzDistYCbCr(E, D)));
        res = mix(res, blendPix, xbrzLeftRatio(pos, origin, direction, scale));
    }

    return vec4(res, 1.0);
}

void main() {
    if (scalingFilter == 2) {
        gl_FragColor = refinedScale();
    } else if (scalingFilter == 3) {
        gl_FragColor = xbrzFreescale();
    } else {
        gl_FragColor = texture2D(frameTexture, vTexCoord);
    }
}
)";

} // namespace

EmulatorView::EmulatorView(QWidget *parent)
    : QOpenGLWidget(parent)
    , vertex_buffer_(QOpenGLBuffer::VertexBuffer) {
    setMinimumSize(320, 240);
    setFocusPolicy(Qt::StrongFocus);
    fps_timer_.start();
}

EmulatorView::~EmulatorView() {
    makeCurrent();
    if (texture_ != 0) {
        glDeleteTextures(1, &texture_);
        texture_ = 0;
    }
    vertex_array_.destroy();
    vertex_buffer_.destroy();
    doneCurrent();
}

QSize EmulatorView::sizeHint() const {
    return QSize(960, 720);
}

void EmulatorView::setSmoothScaling(bool enabled) {
    setScalingFilter(enabled ? ScalingFilter::Linear : ScalingFilter::Nearest);
}

bool EmulatorView::smoothScaling() const {
    return scaling_filter_ == ScalingFilter::Linear;
}

void EmulatorView::setScalingFilter(ScalingFilter filter) {
    if (scaling_filter_ == filter)
        return;

    scaling_filter_ = filter;
    if (isValid()) {
        makeCurrent();
        updateTextureFiltering();
        doneCurrent();
        update();
    }
}

EmulatorView::ScalingFilter EmulatorView::scalingFilter() const {
    return scaling_filter_;
}

void EmulatorView::setSuper2xSaiParameters(float sharpAmount, float edgeBlend, float nearestHold) {
    sharpAmount = std::clamp(sharpAmount, 0.0f, 1.0f);
    edgeBlend = std::clamp(edgeBlend, 0.0f, 1.0f);
    nearestHold = std::clamp(nearestHold, 0.0f, 1.0f);

    if (super2xsai_sharp_amount_ == sharpAmount &&
        super2xsai_edge_blend_ == edgeBlend &&
        super2xsai_nearest_hold_ == nearestHold) {
        return;
    }

    super2xsai_sharp_amount_ = sharpAmount;
    super2xsai_edge_blend_ = edgeBlend;
    super2xsai_nearest_hold_ = nearestHold;
    update();
}

float EmulatorView::super2xSaiSharpAmount() const {
    return super2xsai_sharp_amount_;
}

float EmulatorView::super2xSaiEdgeBlend() const {
    return super2xsai_edge_blend_;
}

float EmulatorView::super2xSaiNearestHold() const {
    return super2xsai_nearest_hold_;
}

void EmulatorView::submitFrame(const void *pixels, int width, int height, int pitch, PixelFormat format) {
    if (!pixels || width <= 0 || height <= 0)
        return;

    Frame frame;
    frame.width = width;
    frame.height = height;
    frame.format = format;

    const int bytesPerPixel = frame.bytesPerPixel();
    const int rowBytes = width * bytesPerPixel;
    if (pitch < rowBytes)
        return;

    frame.pixels.resize(rowBytes * height);
    auto *destination = frame.pixels.data();
    const auto *source = static_cast<const char *>(pixels);

    for (int y = 0; y < height; ++y) {
        std::memcpy(destination + (y * rowBytes), source + (y * pitch), rowBytes);
    }

    {
        QMutexLocker lock(&frame_mutex_);
        pending_frame_ = std::move(frame);
        has_pending_frame_ = true;
    }

    QMetaObject::invokeMethod(this, [this] { update(); }, Qt::QueuedConnection);
}

void EmulatorView::clearFrame() {
    {
        QMutexLocker lock(&frame_mutex_);
        pending_frame_ = Frame {};
        current_frame_ = Frame {};
        has_pending_frame_ = false;
    }
    fps_frame_count_ = 0;
    fps_ = 0.0;
    fps_timer_.restart();
    emit fpsChanged(0.0);
    update();
}

void EmulatorView::initializeGL() {
    initializeOpenGLFunctions();

    glClearColor(0.015f, 0.015f, 0.018f, 1.0f);

    program_.addShaderFromSourceCode(QOpenGLShader::Vertex, VertexShader);
    program_.addShaderFromSourceCode(QOpenGLShader::Fragment, FragmentShader);
    program_.link();

    glGenTextures(1, &texture_);
    glBindTexture(GL_TEXTURE_2D, texture_);
    updateTextureFiltering();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    vertex_array_.create();
    QOpenGLVertexArrayObject::Binder vertex_array_binder(&vertex_array_);

    vertex_buffer_.create();
    vertex_buffer_.bind();
    vertex_buffer_.setUsagePattern(QOpenGLBuffer::DynamicDraw);
    vertex_buffer_.allocate(4 * 4 * static_cast<int>(sizeof(float)));

    program_.bind();
    const int positionLocation = program_.attributeLocation("position");
    const int texCoordLocation = program_.attributeLocation("texCoord");
    program_.enableAttributeArray(positionLocation);
    program_.setAttributeBuffer(positionLocation, GL_FLOAT, 0, 2, 4 * sizeof(float));
    program_.enableAttributeArray(texCoordLocation);
    program_.setAttributeBuffer(texCoordLocation, GL_FLOAT, 2 * sizeof(float), 2, 4 * sizeof(float));
    program_.setUniformValue("frameTexture", 0);
    program_.setUniformValue("textureSize", QVector2D(1.0f, 1.0f));
    program_.setUniformValue("outputSize", QVector2D(1.0f, 1.0f));
    program_.setUniformValue("scalingFilter", 0);
    program_.setUniformValue("sharpAmount", super2xsai_sharp_amount_);
    program_.setUniformValue("edgeBlend", super2xsai_edge_blend_);
    program_.setUniformValue("nearestHold", super2xsai_nearest_hold_);
    program_.release();

    vertex_buffer_.release();
}

void EmulatorView::resizeGL(int width, int height) {
    glViewport(0, 0, width, height);
}

void EmulatorView::paintGL() {
    Frame frameToUpload;
    {
        QMutexLocker lock(&frame_mutex_);
        if (has_pending_frame_) {
            frameToUpload = std::move(pending_frame_);
            has_pending_frame_ = false;
        }
    }

    if (frameToUpload.isValid()) {
        current_frame_ = std::move(frameToUpload);
        uploadFrame(current_frame_);
    }

    glClear(GL_COLOR_BUFFER_BIT);

    if (!current_frame_.isValid() || texture_ == 0)
        return;

    updateVertices();

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture_);

    program_.bind();
    program_.setUniformValue("textureSize", QVector2D(static_cast<float>(current_frame_.width), static_cast<float>(current_frame_.height)));
    float output_width = static_cast<float>(width());
    float output_height = static_cast<float>(height());
    const float widgetAspect = output_width / output_height;
    const float frameAspect = static_cast<float>(current_frame_.width) / static_cast<float>(current_frame_.height);
    if (widgetAspect > frameAspect) {
        output_width = output_height * frameAspect;
    } else {
        output_height = output_width / frameAspect;
    }
    program_.setUniformValue("outputSize", QVector2D(std::max(1.0f, output_width), std::max(1.0f, output_height)));
    program_.setUniformValue("scalingFilter", static_cast<int>(scaling_filter_));
    program_.setUniformValue("sharpAmount", super2xsai_sharp_amount_);
    program_.setUniformValue("edgeBlend", super2xsai_edge_blend_);
    program_.setUniformValue("nearestHold", super2xsai_nearest_hold_);
    QOpenGLVertexArrayObject::Binder vertex_array_binder(&vertex_array_);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    program_.release();

    updateFpsCounter();
}

bool EmulatorView::Frame::isValid() const {
    return width > 0 && height > 0 && pixels.size() == width * height * bytesPerPixel();
}

int EmulatorView::Frame::bytesPerPixel() const {
    switch (format) {
    case PixelFormat::Rgb565:
        return 2;
    case PixelFormat::Rgba8888:
        return 4;
    }
    return 0;
}

void EmulatorView::uploadFrame(const Frame &frame) {
    if (!frame.isValid() || texture_ == 0)
        return;

    glBindTexture(GL_TEXTURE_2D, texture_);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    const QSize frameSize(frame.width, frame.height);
    const bool texture_changed = frameSize != texture_size_ || frame.format != texture_format_;

    GLenum format = GL_RGB;
    GLenum type = GL_UNSIGNED_SHORT_5_6_5;
    GLint internalFormat = GL_RGB;

    if (frame.format == PixelFormat::Rgba8888) {
        format = GL_RGBA;
        type = GL_UNSIGNED_BYTE;
        internalFormat = GL_RGBA;
    }

    if (texture_changed) {
        texture_size_ = frameSize;
        texture_format_ = frame.format;
        glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, frame.width, frame.height, 0, format, type, frame.pixels.constData());
    } else {
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, frame.width, frame.height, format, type, frame.pixels.constData());
    }
}

void EmulatorView::updateTextureFiltering() {
    if (texture_ == 0)
        return;

    glBindTexture(GL_TEXTURE_2D, texture_);
    const GLint filter = scaling_filter_ == ScalingFilter::Linear ? GL_LINEAR : GL_NEAREST;
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter);
}

void EmulatorView::updateVertices() {
    if (!current_frame_.isValid() || width() <= 0 || height() <= 0)
        return;

    const float widgetAspect = static_cast<float>(width()) / static_cast<float>(height());
    const float frameAspect = static_cast<float>(current_frame_.width) / static_cast<float>(current_frame_.height);

    float x = 1.0f;
    float y = 1.0f;
    if (widgetAspect > frameAspect) {
        x = frameAspect / widgetAspect;
    } else {
        y = widgetAspect / frameAspect;
    }

    const float vertices[] = {
        -x,  y, 0.0f, 0.0f,
        -x, -y, 0.0f, 1.0f,
         x,  y, 1.0f, 0.0f,
         x, -y, 1.0f, 1.0f,
    };

    vertex_buffer_.bind();
    vertex_buffer_.write(0, vertices, static_cast<int>(sizeof(vertices)));
    vertex_buffer_.release();
}

void EmulatorView::updateFpsCounter() {
    ++fps_frame_count_;

    const qint64 elapsed = fps_timer_.elapsed();
    if (elapsed < 1000)
        return;

    fps_ = static_cast<double>(fps_frame_count_) * 1000.0 / static_cast<double>(elapsed);
    fps_frame_count_ = 0;
    fps_timer_.restart();
    emit fpsChanged(fps_);
}
