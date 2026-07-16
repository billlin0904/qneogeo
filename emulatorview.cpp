#include "emulatorview.h"

#include <QCoreApplication>
#include <QDebug>
#include <QFile>
#include <QFont>
#include <QMatrix4x4>
#include <QMetaObject>
#include <QPainter>
#include <QPen>
#include <QVector2D>
#include <QWidget>
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

class HitboxOverlayWidget final : public QWidget {
public:
    explicit HitboxOverlayWidget(QWidget *parent = nullptr)
        : QWidget(parent) {
        setAttribute(Qt::WA_TransparentForMouseEvents);
        setAttribute(Qt::WA_NoSystemBackground);
        setAttribute(Qt::WA_TranslucentBackground);
    }

    void setOverlay(QVector<EmulatorView::HitboxRect> boxes,
                    QVector<EmulatorView::HitboxAxis> axes,
                    QSize sourceSize) {
        boxes_ = std::move(boxes);
        axes_ = std::move(axes);
        source_size_ = sourceSize;
        update();
    }

protected:
    void paintEvent(QPaintEvent *) override {
        if (source_size_.isEmpty())
            return;

        float output_width = static_cast<float>(width());
        float output_height = static_cast<float>(height());
        const float widget_aspect = output_width / output_height;
        const float frame_aspect = static_cast<float>(source_size_.width()) / static_cast<float>(source_size_.height());
        if (widget_aspect > frame_aspect) {
            output_width = output_height * frame_aspect;
        } else {
            output_height = output_width / frame_aspect;
        }

        const float scale = output_width / static_cast<float>(source_size_.width());
        const float offset_x = (static_cast<float>(width()) - output_width) * 0.5f;
        const float offset_y = (static_cast<float>(height()) - output_height) * 0.5f;

        auto mapPoint = [&](const QPointF &point) {
            return QPointF(offset_x + point.x() * scale,
                           offset_y + point.y() * scale);
        };

        auto mapRect = [&](const QRectF &rect) {
            return QRectF(offset_x + rect.x() * scale,
                          offset_y + rect.y() * scale,
                          rect.width() * scale,
                          rect.height() * scale);
        };

        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing, false);

        for (const EmulatorView::HitboxRect &box : boxes_) {
            const QRectF target = mapRect(box.rect);
            painter.fillRect(target, box.fill_color);
            QPen pen(box.outline_color);
            pen.setWidthF(2.0f);
            painter.setPen(pen);
            painter.drawRect(target.adjusted(1.0, 1.0, -1.0, -1.0));
        }

        for (const EmulatorView::HitboxAxis &axis : axes_) {
            const QPointF center = mapPoint(axis.position);
            QPen pen(axis.color);
            pen.setWidthF(std::max(1.0f, scale));
            painter.setPen(pen);
            constexpr qreal axis_size = 12.0;
            painter.drawLine(QPointF(center.x(), center.y() - axis_size * scale),
                             QPointF(center.x(), center.y() + axis_size * scale));
            painter.drawLine(QPointF(center.x() - axis_size * scale, center.y()),
                             QPointF(center.x() + axis_size * scale, center.y()));
        }
    }

private:
    QVector<EmulatorView::HitboxRect> boxes_;
    QVector<EmulatorView::HitboxAxis> axes_;
    QSize source_size_;
};

class InputHistoryOverlayWidget final : public QWidget {
public:
    explicit InputHistoryOverlayWidget(QWidget *parent = nullptr)
        : QWidget(parent) {
        setAttribute(Qt::WA_TransparentForMouseEvents);
        setAttribute(Qt::WA_NoSystemBackground);
        setAttribute(Qt::WA_TranslucentBackground);
    }

    void appendInput(const EmulatorView::JoypadInput &input, uint64_t frameNumber) {
        if (frameNumber == 0)
            return;

        if (!history_.isEmpty() && frameNumber <= history_.constLast().last_frame)
            history_.clear();

        if (!history_.isEmpty() &&
            history_.constLast().input == input &&
            frameNumber == history_.constLast().last_frame + 1) {
            history_.last().last_frame = frameNumber;
        } else {
            history_.push_back({ input, frameNumber, frameNumber });
            while (history_.size() > MAX_HISTORY_ENTRIES)
                history_.remove(0);
        }

        update();
    }

    void clearHistory() {
        history_.clear();
        update();
    }

protected:
    void paintEvent(QPaintEvent *) override {
        if (history_.isEmpty())
            return;

        constexpr int32_t PANEL_X = 12;
        constexpr int32_t PANEL_TOP = 52;
        constexpr int32_t PANEL_WIDTH = 252;
        constexpr int32_t HEADER_HEIGHT = 24;
        constexpr int32_t ROW_HEIGHT = 22;
        constexpr int32_t PANEL_MARGIN_BOTTOM = 12;

        const int32_t available_height = height() - PANEL_TOP - HEADER_HEIGHT - PANEL_MARGIN_BOTTOM;
        const int32_t visible_count = std::min<int32_t>(
            static_cast<int32_t>(history_.size()),
            std::max<int32_t>(1, available_height / ROW_HEIGHT));
        const int32_t first_visible = static_cast<int32_t>(history_.size()) - visible_count;
        const QRect panel_rect(PANEL_X,
                               PANEL_TOP,
                               PANEL_WIDTH,
                               HEADER_HEIGHT + visible_count * ROW_HEIGHT);

        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing, false);
        painter.fillRect(panel_rect, QColor(0, 0, 0, 176));
        painter.setPen(QPen(QColor(255, 255, 255, 100), 1));
        painter.drawRect(panel_rect.adjusted(0, 0, -1, -1));

        QFont font(QStringLiteral("SF Mono"));
        font.setStyleHint(QFont::Monospace);
        font.setPixelSize(12);
        font.setBold(true);
        painter.setFont(font);
        painter.setPen(Qt::white);
        painter.drawText(QRect(PANEL_X + 8, PANEL_TOP, PANEL_WIDTH - 16, HEADER_HEIGHT),
                         Qt::AlignVCenter | Qt::AlignLeft,
                         QStringLiteral("P1 INPUT / FRAME"));

        for (int32_t row = 0; row < visible_count; ++row) {
            const Entry &entry = history_.at(first_visible + row);
            const int32_t row_y = PANEL_TOP + HEADER_HEIGHT + row * ROW_HEIGHT;
            if ((row & 1) != 0)
                painter.fillRect(QRect(PANEL_X + 1, row_y, PANEL_WIDTH - 2, ROW_HEIGHT),
                                 QColor(255, 255, 255, 12));

            painter.setPen(QColor(225, 225, 225));
            painter.drawText(QRect(PANEL_X + 8, row_y, 108, ROW_HEIGHT),
                             Qt::AlignVCenter | Qt::AlignLeft,
                             frameText(entry));
            painter.setPen(Qt::white);
            painter.drawText(QRect(PANEL_X + 116, row_y, 30, ROW_HEIGHT),
                             Qt::AlignCenter,
                             directionText(entry.input));

            drawButton(painter, PANEL_X + 150, row_y + 2, QStringLiteral("A"), entry.input.a,
                       QColor(225, 56, 54), Qt::white);
            drawButton(painter, PANEL_X + 173, row_y + 2, QStringLiteral("B"), entry.input.b,
                       QColor(250, 193, 31), Qt::black);
            drawButton(painter, PANEL_X + 196, row_y + 2, QStringLiteral("C"), entry.input.c,
                       QColor(30, 196, 91), Qt::black);
            drawButton(painter, PANEL_X + 219, row_y + 2, QStringLiteral("D"), entry.input.d,
                       QColor(55, 125, 230), Qt::white);
        }
    }

private:
    struct Entry {
        EmulatorView::JoypadInput input;
        uint64_t first_frame = 0;
        uint64_t last_frame = 0;
    };

    static constexpr int32_t MAX_HISTORY_ENTRIES = 18;

    static QString frameText(const Entry &entry) {
        const QString first = QStringLiteral("%1")
                                  .arg(static_cast<qulonglong>(entry.first_frame), 6, 10, QLatin1Char('0'));
        if (entry.first_frame == entry.last_frame)
            return QStringLiteral("F%1").arg(first);

        const QString last = QStringLiteral("%1")
                                 .arg(static_cast<qulonglong>(entry.last_frame), 6, 10, QLatin1Char('0'));
        return QStringLiteral("F%1-%2").arg(first, last);
    }

    static QString directionText(const EmulatorView::JoypadInput &input) {
        if (input.left && input.right)
            return QStringLiteral("L+R");
        if (input.up && input.down)
            return QStringLiteral("U+D");
        if (input.up && input.left)
            return QStringLiteral("↖");
        if (input.up && input.right)
            return QStringLiteral("↗");
        if (input.down && input.left)
            return QStringLiteral("↙");
        if (input.down && input.right)
            return QStringLiteral("↘");
        if (input.up)
            return QStringLiteral("↑");
        if (input.down)
            return QStringLiteral("↓");
        if (input.left)
            return QStringLiteral("←");
        if (input.right)
            return QStringLiteral("→");
        return QStringLiteral("-");
    }

    static void drawButton(QPainter &painter,
                           int32_t x,
                           int32_t y,
                           const QString &label,
                           bool active,
                           const QColor &fill,
                           const QColor &textColor) {
        if (!active)
            return;

        const QRect rect(x, y, 19, 18);
        painter.fillRect(rect, fill);
        painter.setPen(QPen(fill.lighter(135), 1));
        painter.drawRect(rect.adjusted(0, 0, -1, -1));
        painter.setPen(textColor);
        painter.drawText(rect, Qt::AlignCenter, label);
    }

    QVector<Entry> history_;
};

} // namespace

EmulatorView::EmulatorView(QWidget *parent)
    : QOpenGLWidget(parent)
    , vertex_buffer_(QOpenGLBuffer::VertexBuffer) {
    setMinimumSize(320, 240);
    setFocusPolicy(Qt::StrongFocus);
    hitbox_overlay_widget_ = new HitboxOverlayWidget(this);
    hitbox_overlay_widget_->setGeometry(rect());
    hitbox_overlay_widget_->raise();
    hitbox_overlay_widget_->hide();
    input_overlay_widget_ = new InputHistoryOverlayWidget(this);
    input_overlay_widget_->setGeometry(rect());
    input_overlay_widget_->raise();
    input_overlay_widget_->show();
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
    zfast_crt_program_.removeAllShaders();
    zfast_lcd_program_.removeAllShaders();
    scanline_fract_program_.removeAllShaders();
    vertex_array_.destroy();
    vertex_buffer_.destroy();
    doneCurrent();
}

QSize EmulatorView::sizeHint() const {
    return QSize(960, 720);
}

QSize EmulatorView::sourceSize() const {
    QMutexLocker lock(&frame_mutex_);
    return QSize(current_frame_.width, current_frame_.height);
}

void EmulatorView::setSmoothScaling(bool enabled) {
    setScalingFilter(enabled ? ScalingFilter::Linear : ScalingFilter::Nearest);
}

bool EmulatorView::smoothScaling() const {
    return scaling_filter_ == ScalingFilter::Linear;
}

void EmulatorView::setHitboxOverlayEnabled(bool enabled) {
    if (hitbox_overlay_enabled_ == enabled)
        return;

    hitbox_overlay_enabled_ = enabled;
    if (hitbox_overlay_widget_) {
        hitbox_overlay_widget_->setVisible(enabled);
        if (enabled)
            hitbox_overlay_widget_->raise();
    }
    update();
}

bool EmulatorView::hitboxOverlayEnabled() const {
    return hitbox_overlay_enabled_;
}

void EmulatorView::setInputOverlayEnabled(bool enabled) {
    if (input_overlay_enabled_ == enabled)
        return;

    input_overlay_enabled_ = enabled;
    if (input_overlay_widget_) {
        input_overlay_widget_->setVisible(enabled);
        if (enabled)
            input_overlay_widget_->raise();
    }
}

bool EmulatorView::inputOverlayEnabled() const {
    return input_overlay_enabled_;
}

void EmulatorView::setHitboxOverlay(QVector<HitboxRect> boxes, QVector<HitboxAxis> axes) {
    hitbox_boxes_ = std::move(boxes);
    hitbox_axes_ = std::move(axes);
    updateHitboxOverlayWidget();
    if (hitbox_overlay_enabled_)
        update();
}

bool EmulatorView::JoypadInput::operator==(const JoypadInput &other) const {
    return up == other.up &&
           down == other.down &&
           left == other.left &&
           right == other.right &&
           a == other.a &&
           b == other.b &&
           c == other.c &&
           d == other.d;
}

bool EmulatorView::JoypadInput::isNeutral() const {
    return !up && !down && !left && !right && !a && !b && !c && !d;
}

void EmulatorView::submitInputFrame(const JoypadInput &input, uint64_t frameNumber) {
    auto *overlay = static_cast<InputHistoryOverlayWidget *>(input_overlay_widget_);
    if (!overlay)
        return;

    overlay->appendInput(input, frameNumber);
    if (input_overlay_enabled_) {
        overlay->show();
        overlay->raise();
    }
}

void EmulatorView::clearInputHistory() {
    auto *overlay = static_cast<InputHistoryOverlayWidget *>(input_overlay_widget_);
    if (overlay)
        overlay->clearHistory();
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
    clearInputHistory();
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
    initializeLibretroShader(zfast_crt_program_, QStringLiteral("zfast_crt.glsl"));
    initializeLibretroShader(zfast_lcd_program_, QStringLiteral("zfast_lcd.glsl"));
    initializeLibretroShader(scanline_fract_program_, QStringLiteral("scanline-fract.glsl"));

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
    program_.release();

    vertex_buffer_.release();
}

void EmulatorView::resizeGL(int width, int height) {
    glViewport(0, 0, width, height);
    if (hitbox_overlay_widget_)
        hitbox_overlay_widget_->setGeometry(rect());
    if (input_overlay_widget_)
        input_overlay_widget_->setGeometry(rect());
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
        updateHitboxOverlayWidget();
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
    } else if (scaling_filter_ == ScalingFilter::ZfastCrt && zfast_crt_program_.isLinked()) {
        active_program = &zfast_crt_program_;
    } else if (scaling_filter_ == ScalingFilter::ZfastLcd && zfast_lcd_program_.isLinked()) {
        active_program = &zfast_lcd_program_;
    } else if (scaling_filter_ == ScalingFilter::ScanlineFract && scanline_fract_program_.isLinked()) {
        active_program = &scanline_fract_program_;
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
           (scaling_filter_ == ScalingFilter::Libretro6xbrz && libretro_6xbrz_program_.isLinked()) ||
           (scaling_filter_ == ScalingFilter::ZfastCrt && zfast_crt_program_.isLinked()) ||
           (scaling_filter_ == ScalingFilter::ZfastLcd && zfast_lcd_program_.isLinked()) ||
           (scaling_filter_ == ScalingFilter::ScanlineFract && scanline_fract_program_.isLinked());
}

void EmulatorView::updateHitboxOverlayWidget() {
    auto *overlay = static_cast<HitboxOverlayWidget *>(hitbox_overlay_widget_);
    if (!overlay)
        return;

    const QSize source_size(current_frame_.width, current_frame_.height);
    overlay->setGeometry(rect());
    if (hitbox_overlay_enabled_) {
        overlay->show();
        overlay->raise();
    }
    overlay->setOverlay(hitbox_boxes_, hitbox_axes_, source_size);

    if (input_overlay_enabled_ && input_overlay_widget_)
        input_overlay_widget_->raise();
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
