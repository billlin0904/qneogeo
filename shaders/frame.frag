uniform sampler2D frameTexture;
uniform vec2 textureSize;
uniform vec2 outputSize;
uniform int scalingFilter;
varying vec2 vTexCoord;

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
    if (scalingFilter == 3) {
        gl_FragColor = xbrzFreescale();
    } else {
        gl_FragColor = texture2D(frameTexture, vTexCoord);
    }
}
