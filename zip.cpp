#include <unzip.h>
#include "zip.h"

static voidpf ZCALLBACK fopen_file_func(voidpf opaque, const char* filename, int mode) {
    FILE* file = nullptr;
    const char* mode_fopen = nullptr;
    if ((mode & ZLIB_FILEFUNC_MODE_READWRITEFILTER)==ZLIB_FILEFUNC_MODE_READ)
        mode_fopen = "rb";
    else
        if (mode & ZLIB_FILEFUNC_MODE_EXISTING)
        mode_fopen = "r+b";
    else
        if (mode & ZLIB_FILEFUNC_MODE_CREATE)
        mode_fopen = "wb";

    if ((filename != nullptr) && (mode_fopen != nullptr))
        file = fopen(filename, mode_fopen);
    return file;
}

static uLong ZCALLBACK fread_file_func(voidpf opaque, voidpf stream, void* buf, uLong size) {
    return static_cast<uLong>(fread(buf, 1, size_t(size), reinterpret_cast<FILE*>(stream)));
}

static uLong ZCALLBACK fwrite_file_func(voidpf opaque, voidpf stream, const void* buf, uLong size) {
    return static_cast<uLong>(fwrite(buf, 1, size_t(size), reinterpret_cast<FILE*>(stream)));
}

static long ZCALLBACK ftell_file_func(voidpf opaque, voidpf stream) {
    return static_cast<long>(ftell(reinterpret_cast<FILE*>(stream)));
}

static long ZCALLBACK fseek_file_func(voidpf opaque, voidpf stream, uLong offset, int origin) {
    int fseek_origin = 0;
    long ret;

    switch (origin) {
    case ZLIB_FILEFUNC_SEEK_CUR :
        fseek_origin = SEEK_CUR;
        break;
    case ZLIB_FILEFUNC_SEEK_END :
        fseek_origin = SEEK_END;
        break;
    case ZLIB_FILEFUNC_SEEK_SET :
        fseek_origin = SEEK_SET;
        break;
    default: return -1;
    }

    ret = 0;
    if (fseek(reinterpret_cast<FILE*>(stream), offset, fseek_origin) != 0) {
        return -1;
    }
    return ret;
}

static int ZCALLBACK fclose_file_func(voidpf opaque, voidpf stream) {
    return fclose(reinterpret_cast<FILE*>(stream));
}

static int ZCALLBACK ferror_file_func(voidpf opaque, voidpf stream) {
    return ferror(reinterpret_cast<FILE*>(stream));
}

class Zip {
public:
    explicit Zip(const std::string& archiveFilename) {
        callbacks_.zopen_file = fopen_file_func;
        callbacks_.zread_file = fread_file_func;
        callbacks_.zwrite_file = fwrite_file_func;
        callbacks_.ztell_file = ftell_file_func;
        callbacks_.zseek_file = fseek_file_func;
        callbacks_.zclose_file = fclose_file_func;
        callbacks_.zerror_file = ferror_file_func;
        callbacks_.opaque = nullptr;
        open(archiveFilename);
    }

    void open(const std::string& archiveFilename) {
        file_ = ::unzOpen2(archiveFilename.c_str(), &callbacks_);
        if (!file_) {
            return;
        }
    }

    unz_file_info getFileInfo() {
        unz_file_info fileInfo {};
        if (::unzGetCurrentFileInfo(file_, &fileInfo, nullptr, 0, nullptr, 0, nullptr, 0) != UNZ_OK) {
        }
        return fileInfo;
    }

    unz_file_info getFileInfo(unz_file_info fileInfo) {
        std::string filename(fileInfo.size_filename, 0x0);
        if (::unzGetCurrentFileInfo(file_, &fileInfo, const_cast<char *>(filename.data()), filename.size(), nullptr, 0, nullptr, 0) != UNZ_OK) {
        }
        return fileInfo;
    }

    bool nextFile() {
        return ::unzGoToNextFile(file_) != UNZ_OK;
    }

private:
    zlib_filefunc_def callbacks_;
    unzFile file_;
};

std::vector<std::string> getFileList(const std::string& archiveFilename) {
    std::vector<std::string> result;
    return result;
}
