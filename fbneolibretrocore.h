#pragma once

#include "libretrocore.h"

class FbneoLibretroCore final : public LibretroCore {
    Q_OBJECT

public:
    explicit FbneoLibretroCore(EmulatorView *videoOutput, QObject *parent = nullptr);

    QString displayName() const override;
    QString coreFileName() const override;
    QString romDirectoryName() const override;
    QStringList supportedExtensions() const override;

protected:
    bool coreOptionValue(const QByteArray &key, const char *&value) const override;

private:
    mutable std::string neogeo_mode_value_;
};
