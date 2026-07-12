#pragma once

#include <QByteArray>
#include <QColor>
#include <QElapsedTimer>
#include <QMutex>
#include <QOpenGLBuffer>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLWidget>
#include <QPointF>
#include <QRectF>
#include <QVector>

class EmulatorView final : public QOpenGLWidget, protected QOpenGLFunctions {
    Q_OBJECT

public:
    enum class PixelFormat {
        Rgb565,
        Rgba8888
    };

    enum class ScalingFilter {
        Nearest = 0,
        Linear = 1,
        XbrzFreescale = 3,
        LibretroXbrzFreescale = 4,
        Libretro6xbrz = 5,
        ZfastCrt = 6,
        ZfastLcd = 7,
        ScanlineFract = 8
    };

    explicit EmulatorView(QWidget *parent = nullptr);
    ~EmulatorView() override;

    QSize sizeHint() const override;
    QSize sourceSize() const;

    void setScalingFilter(ScalingFilter filter);
    ScalingFilter scalingFilter() const;
    void setSmoothScaling(bool enabled);
    bool smoothScaling() const;
    void setHitboxOverlayEnabled(bool enabled);
    bool hitboxOverlayEnabled() const;

    struct HitboxRect {
        QRectF rect;
        QColor fill_color;
        QColor outline_color;
    };

    struct HitboxAxis {
        QPointF position;
        QColor color;
    };

    void setHitboxOverlay(QVector<HitboxRect> boxes, QVector<HitboxAxis> axes);

signals:
    void fpsChanged(double fps);

public slots:
    void submitFrame(const void *pixels, int width, int height, int pitch, PixelFormat format);
    void clearFrame();

protected:
    void initializeGL() override;
    void resizeGL(int width, int height) override;
    void paintGL() override;

private:
    struct Frame {
        QByteArray pixels;
        int width = 0;
        int height = 0;
        PixelFormat format = PixelFormat::Rgb565;

        bool isValid() const;
        int bytesPerPixel() const;
    };

    void uploadFrame(const Frame &frame);
    void updateTextureFiltering();
    void updateVertices();
    void updateFpsCounter();
    bool initializeFrameShader();
    bool initializeLibretroShader(QOpenGLShaderProgram &program, const QString &fileName);
    bool usesLibretroShader() const;
    void updateHitboxOverlayWidget();

    mutable QMutex frame_mutex_;
    Frame pending_frame_;
    bool has_pending_frame_ = false;
    bool shutting_down_ = false;

    Frame current_frame_;
    QSize texture_size_;
    PixelFormat texture_format_ = PixelFormat::Rgb565;
    GLuint texture_ = 0;
    ScalingFilter scaling_filter_ = ScalingFilter::Nearest;
    QElapsedTimer fps_timer_;
    int fps_frame_count_ = 0;
    double fps_ = 0.0;
    bool hitbox_overlay_enabled_ = false;
    QVector<HitboxRect> hitbox_boxes_;
    QVector<HitboxAxis> hitbox_axes_;
    QWidget *hitbox_overlay_widget_ = nullptr;

    QOpenGLShaderProgram program_;
    QOpenGLShaderProgram libretro_xbrz_freescale_program_;
    QOpenGLShaderProgram libretro_6xbrz_program_;
    QOpenGLShaderProgram zfast_crt_program_;
    QOpenGLShaderProgram zfast_lcd_program_;
    QOpenGLShaderProgram scanline_fract_program_;
    QOpenGLBuffer vertex_buffer_;
    QOpenGLVertexArrayObject vertex_array_;
};
