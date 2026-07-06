#include "neocdlibretrocore.h"

NeoCdLibretroCore::NeoCdLibretroCore(EmulatorView *videoOutput, QObject *parent)
    : LibretroCore(videoOutput, parent) {
}

QString NeoCdLibretroCore::displayName() const {
    return QStringLiteral("Neo Geo CD");
}

QString NeoCdLibretroCore::coreFileName() const {
    return QStringLiteral("neocd_libretro.dll");
}

QString NeoCdLibretroCore::romDirectoryName() const {
    return QStringLiteral("neocd");
}

QStringList NeoCdLibretroCore::supportedExtensions() const {
    return {
        QStringLiteral("cue"),
        QStringLiteral("chd")
    };
}

bool NeoCdLibretroCore::coreOptionValue(const QByteArray &key, const char *&value) const {
    if (key == "neocd_region") {
        value = system_region_option_.c_str();
        return true;
    }
    if (key == "neocd_cdspeedhack") {
        value = "On";
        return true;
    }
    if (key == "neocd_loadskip") {
        value = "On";
        return true;
    }
    if (key == "neocd_per_content_saves") {
        value = "Off";
        return true;
    }
    if (key == "neocd_bios" && !selected_bios_.empty()) {
        value = selected_bios_.c_str();
        return true;
    }

    return false;
}

void NeoCdLibretroCore::coreOptionsUpdated(const retro_core_options_v2 *options) {
    selected_bios_.clear();

    for (const retro_core_option_v2_definition *definition = options->definitions; definition->key; ++definition) {
        if (QByteArray(definition->key) != "neocd_bios")
            continue;

        if (definition->default_value) {
            selected_bios_ = definition->default_value;
        } else if (definition->values[0].value) {
            selected_bios_ = definition->values[0].value;
        }
        return;
    }
}
