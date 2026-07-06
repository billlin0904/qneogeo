#include "fbneolibretrocore.h"

FbneoLibretroCore::FbneoLibretroCore(EmulatorView *videoOutput, QObject *parent)
    : LibretroCore(videoOutput, parent) {
}

QString FbneoLibretroCore::displayName() const {
    return QStringLiteral("Neo Geo Arcade (FBNeo)");
}

QString FbneoLibretroCore::coreFileName() const {
    return QStringLiteral("fbneo_libretro.dll");
}

QString FbneoLibretroCore::romDirectoryName() const {
    return QStringLiteral("fbneo");
}

QStringList FbneoLibretroCore::supportedExtensions() const {
    return {
        QStringLiteral("zip"),
        QStringLiteral("7z")
    };
}

bool FbneoLibretroCore::coreOptionValue(const QByteArray &key, const char *&value) const {
    if (key == "fbneo-cpu-speed-adjust") {
        value = fbneo_cpu_clock_option_.c_str();
        return true;
    }
    if (key == "fbneo-frameskip") {
        value = "0";
        return true;
    }
    if (key == "fbneo-lightgun-hide-crosshair") {
        value = "enabled";
        return true;
    }
    if (key == "fbneo-neogeo-mode" || key == "fbneo-neogeo-mode-switch") {
        if (system_mode_option_ == "UNIBIOS" || system_mode_option_ == "DIPSWITCH") {
            neogeo_mode_value_ = system_mode_option_;
        } else if (system_mode_option_ == "AES") {
            neogeo_mode_value_ = system_region_option_ == "Japan" ? "AES_JAP" : "AES_EUR";
        } else {
            if (system_region_option_ == "Japan") {
                neogeo_mode_value_ = "MVS_JAP";
            } else if (system_region_option_ == "USA") {
                neogeo_mode_value_ = "MVS_USA";
            } else {
                neogeo_mode_value_ = "MVS_EUR";
            }
        }

        value = neogeo_mode_value_.c_str();
        return true;
    }
    if (key == "fbneo-memcard-mode") {
        value = "shared";
        return true;
    }

    return false;
}
