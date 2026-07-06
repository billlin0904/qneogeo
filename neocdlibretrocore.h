#pragma once

#include "libretrocore.h"

class NeoCdLibretroCore final : public LibretroCore {
    Q_OBJECT

public:
    explicit NeoCdLibretroCore(EmulatorView *videoOutput, QObject *parent = nullptr);

    QString displayName() const override;
    QString coreFileName() const override;
    QString romDirectoryName() const override;
    QStringList supportedExtensions() const override;

protected:
    bool coreOptionValue(const QByteArray &key, const char *&value) const override;
    void coreOptionsUpdated(const retro_core_options_v2 *options) override;
};
