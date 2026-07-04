#pragma once

#include <QByteArray>
#include <QElapsedTimer>
#include <QMutex>
#include <QOpenGLBuffer>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLWidget>

class EmulatorView final : public QOpenGLWidget, protected QOpenGLFunctions {
    Q_OBJECT

public:
    enum class PixelFormat {
        Rgb565,
        Rgba8888
    };

    enum class ScalingFilter {
        Nearest,
        Linear,
        Super2xSai,
        XbrzFreescale
    };

    explicit EmulatorView(QWidget *parent = nullptr);
    ~EmulatorView() override;

    QSize sizeHint() const override;

    void setScalingFilter(ScalingFilter filter);
    ScalingFilter scalingFilter() const;
    void setSuper2xSaiParameters(float sharpAmount, float edgeBlend, float nearestHold);
    float super2xSaiSharpAmount() const;
    float super2xSaiEdgeBlend() const;
    float super2xSaiNearestHold() const;
    void setSmoothScaling(bool enabled);
    bool smoothScaling() const;

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

    QMutex frame_mutex_;
    Frame pending_frame_;
    bool has_pending_frame_ = false;

    Frame current_frame_;
    QSize texture_size_;
    PixelFormat texture_format_ = PixelFormat::Rgb565;
    GLuint texture_ = 0;
    ScalingFilter scaling_filter_ = ScalingFilter::Nearest;
    float super2xsai_sharp_amount_ = 0.25f;
    float super2xsai_edge_blend_ = 0.45f;
    float super2xsai_nearest_hold_ = 0.10f;
    QElapsedTimer fps_timer_;
    int fps_frame_count_ = 0;
    double fps_ = 0.0;

    QOpenGLShaderProgram program_;
    QOpenGLBuffer vertex_buffer_;
    QOpenGLVertexArrayObject vertex_array_;
};
