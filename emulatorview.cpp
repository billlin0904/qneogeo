#include "emulatorview.h"

#include <QCoreApplication>
#include <QDebug>
#include <QFile>
#include <QMatrix4x4>
#include <QMetaObject>
#include <QVector2D>
#include <algorithm>
#include <cstring>

namespace {

QString loadShaderSource(const QString &fileName) {
    const QString path = QCoreApplication::applicationDirPath() + QStringLiteral("/shaders/") + fileName;
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qWarning().noquote() << "Unable to load shader:" << path << file.errorString();
        return {};
    }

    return QString::fromUtf8(file.readAll());
}

QString withShaderStageDefine(const QString &source, const char *stage) {
    return QStringLiteral("#define ") + QString::fromLatin1(stage) + QStringLiteral("\n") + source;
}

} // namespace

EmulatorView::EmulatorView(QWidget *parent)
    : QOpenGLWidget(parent)
    , vertex_buffer_(QOpenGLBuffer::VertexBuffer) {
    setMinimumSize(320, 240);
    setFocusPolicy(Qt::StrongFocus);
    fps_timer_.start();
}

EmulatorView::~EmulatorView() {
    shutting_down_ = true;

    if (!isValid())
        return;

    makeCurrent();
    if (texture_ != 0) {
        glDeleteTextures(1, &texture_);
        texture_ = 0;
    }
    program_.removeAllShaders();
    libretro_xbrz_freescale_program_.removeAllShaders();
    libretro_6xbrz_program_.removeAllShaders();
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
    if (shutting_down_)
        return;

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

    QMetaObject::invokeMethod(this, [this] {
        if (!shutting_down_)
            update();
    }, Qt::QueuedConnection);
}

void EmulatorView::clearFrame() {
    if (shutting_down_)
        return;

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

bool EmulatorView::initializeFrameShader() {
    const QString vertex_shader = loadShaderSource(QStringLiteral("frame.vert"));
    const QString fragment_shader = loadShaderSource(QStringLiteral("frame.frag"));
    if (vertex_shader.isEmpty() || fragment_shader.isEmpty())
        return false;

    if (!program_.addShaderFromSourceCode(QOpenGLShader::Vertex, vertex_shader))
        qWarning().noquote() << "Vertex shader compile failed:" << program_.log();
    if (!program_.addShaderFromSourceCode(QOpenGLShader::Fragment, fragment_shader))
        qWarning().noquote() << "Fragment shader compile failed:" << program_.log();

    program_.bindAttributeLocation("position", 0);
    program_.bindAttributeLocation("texCoord", 1);
    if (!program_.link()) {
        qWarning().noquote() << "Shader link failed:" << program_.log();
        return false;
    }

    return true;
}

bool EmulatorView::initializeLibretroShader(QOpenGLShaderProgram &program, const QString &fileName) {
    const QString shader_source = loadShaderSource(fileName);
    if (shader_source.isEmpty())
        return false;

    if (!program.addShaderFromSourceCode(QOpenGLShader::Vertex, withShaderStageDefine(shader_source, "VERTEX")))
        qWarning().noquote() << fileName << "vertex shader compile failed:" << program.log();
    if (!program.addShaderFromSourceCode(QOpenGLShader::Fragment, withShaderStageDefine(shader_source, "FRAGMENT")))
        qWarning().noquote() << fileName << "fragment shader compile failed:" << program.log();

    program.bindAttributeLocation("VertexCoord", 0);
    program.bindAttributeLocation("TexCoord", 1);
    if (!program.link()) {
        qWarning().noquote() << fileName << "shader link failed:" << program.log();
        return false;
    }

    return true;
}

void EmulatorView::initializeGL() {
    initializeOpenGLFunctions();

    glClearColor(0.015f, 0.015f, 0.018f, 1.0f);

    initializeFrameShader();
    initializeLibretroShader(libretro_xbrz_freescale_program_, QStringLiteral("xbrz-freescale.glsl"));
    initializeLibretroShader(libretro_6xbrz_program_, QStringLiteral("6xbrz.glsl"));

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
    constexpr int positionLocation = 0;
    constexpr int texCoordLocation = 1;
    glEnableVertexAttribArray(positionLocation);
    glVertexAttribPointer(positionLocation, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), nullptr);
    glEnableVertexAttribArray(texCoordLocation);
    glVertexAttribPointer(texCoordLocation, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), reinterpret_cast<void *>(2 * sizeof(float)));
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

    float output_width = static_cast<float>(width());
    float output_height = static_cast<float>(height());
    const float widgetAspect = output_width / output_height;
    const float frameAspect = static_cast<float>(current_frame_.width) / static_cast<float>(current_frame_.height);
    if (widgetAspect > frameAspect) {
        output_width = output_height * frameAspect;
    } else {
        output_height = output_width / frameAspect;
    }

    QOpenGLShaderProgram *active_program = &program_;
    if (scaling_filter_ == ScalingFilter::LibretroXbrzFreescale && libretro_xbrz_freescale_program_.isLinked()) {
        active_program = &libretro_xbrz_freescale_program_;
    } else if (scaling_filter_ == ScalingFilter::Libretro6xbrz && libretro_6xbrz_program_.isLinked()) {
        active_program = &libretro_6xbrz_program_;
    }

    active_program->bind();
    const QVector2D texture_size(static_cast<float>(current_frame_.width), static_cast<float>(current_frame_.height));
    const QVector2D output_size(std::max(1.0f, output_width), std::max(1.0f, output_height));

    if (usesLibretroShader()) {
        QMatrix4x4 mvp_matrix;
        mvp_matrix.setToIdentity();
        active_program->setUniformValue("Texture", 0);
        active_program->setUniformValue("TextureSize", texture_size);
        active_program->setUniformValue("InputSize", texture_size);
        active_program->setUniformValue("OutputSize", output_size);
        active_program->setUniformValue("MVPMatrix", mvp_matrix);
        active_program->setUniformValue("FrameDirection", 0);
        active_program->setUniformValue("FrameCount", fps_frame_count_);
    } else {
        active_program->setUniformValue("frameTexture", 0);
        active_program->setUniformValue("textureSize", texture_size);
        active_program->setUniformValue("outputSize", output_size);
        active_program->setUniformValue("scalingFilter", static_cast<int>(scaling_filter_));
        active_program->setUniformValue("sharpAmount", super2xsai_sharp_amount_);
        active_program->setUniformValue("edgeBlend", super2xsai_edge_blend_);
        active_program->setUniformValue("nearestHold", super2xsai_nearest_hold_);
    }

    QOpenGLVertexArrayObject::Binder vertex_array_binder(&vertex_array_);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    active_program->release();

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

bool EmulatorView::usesLibretroShader() const {
    return (scaling_filter_ == ScalingFilter::LibretroXbrzFreescale && libretro_xbrz_freescale_program_.isLinked()) ||
           (scaling_filter_ == ScalingFilter::Libretro6xbrz && libretro_6xbrz_program_.isLinked());
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
