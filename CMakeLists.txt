cmake_minimum_required(VERSION 3.21)

project(qneogeo LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

option(QNEOGEO_BUILD_QT_APP "Build the Qt shell application" ON)
option(QNEOGEO_BUILD_NEOCD_CORE "Build neocd_libretro as a Visual Studio target" ON)
option(QNEOGEO_BUILD_TOOLS "Build local diagnostic tools" ON)
option(QNEOGEO_FETCH_FBNEO_CORE "Download the Windows x64 fbneo_libretro core from the libretro buildbot" ON)

if(QNEOGEO_BUILD_NEOCD_CORE)
    set(NEOCD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/neocd_libretro")
    set(NEOCD_DEPS_DIR "${NEOCD_DIR}/deps")
    set(LIBRETRO_COMMON_DIR "${NEOCD_DEPS_DIR}/libretro-common")
    set(LIBCHDR_DIR "${NEOCD_DEPS_DIR}/libchdr")
    set(MINIZIP_DIR "${NEOCD_DEPS_DIR}/minizip")
    set(LZMA_DIR "${LIBCHDR_DIR}/deps/lzma-24.05")

    find_package(ZLIB REQUIRED)
    find_package(zstd CONFIG REQUIRED)
    find_package(Ogg CONFIG REQUIRED)
    find_package(Vorbis CONFIG REQUIRED)

    set(NEOCD_CXX_SOURCES
        "${NEOCD_DIR}/src/archive.cpp"
        "${NEOCD_DIR}/src/archivezip.cpp"
        "${NEOCD_DIR}/src/audio.cpp"
        "${NEOCD_DIR}/src/bios.cpp"
        "${NEOCD_DIR}/src/cdrom.cpp"
        "${NEOCD_DIR}/src/cdromcontroller.cpp"
        "${NEOCD_DIR}/src/cdromtoc.cpp"
        "${NEOCD_DIR}/src/chdfile.cpp"
        "${NEOCD_DIR}/src/datapacker.cpp"
        "${NEOCD_DIR}/src/file.cpp"
        "${NEOCD_DIR}/src/flacfile.cpp"
        "${NEOCD_DIR}/src/input.cpp"
        "${NEOCD_DIR}/src/lc8951.cpp"
        "${NEOCD_DIR}/src/libretro_backupram.cpp"
        "${NEOCD_DIR}/src/libretro_bios.cpp"
        "${NEOCD_DIR}/src/libretro_common.cpp"
        "${NEOCD_DIR}/src/libretro_input.cpp"
        "${NEOCD_DIR}/src/libretro_log.cpp"
        "${NEOCD_DIR}/src/libretro_memmap.cpp"
        "${NEOCD_DIR}/src/libretro_variables.cpp"
        "${NEOCD_DIR}/src/libretro.cpp"
        "${NEOCD_DIR}/src/m68kintf.cpp"
        "${NEOCD_DIR}/src/memory_backupram.cpp"
        "${NEOCD_DIR}/src/memory_cdintf.cpp"
        "${NEOCD_DIR}/src/memory_input.cpp"
        "${NEOCD_DIR}/src/memory_mapped.cpp"
        "${NEOCD_DIR}/src/memory_paletteram.cpp"
        "${NEOCD_DIR}/src/memory_switches.cpp"
        "${NEOCD_DIR}/src/memory_video.cpp"
        "${NEOCD_DIR}/src/memory_z80comm.cpp"
        "${NEOCD_DIR}/src/memory.cpp"
        "${NEOCD_DIR}/src/neogeocd.cpp"
        "${NEOCD_DIR}/src/oggfile.cpp"
        "${NEOCD_DIR}/src/path.cpp"
        "${NEOCD_DIR}/src/timeprofiler.cpp"
        "${NEOCD_DIR}/src/timer.cpp"
        "${NEOCD_DIR}/src/timergroup.cpp"
        "${NEOCD_DIR}/src/video.cpp"
        "${NEOCD_DIR}/src/wavfile.cpp"
        "${NEOCD_DIR}/src/z80intf.cpp"
        "${NEOCD_DIR}/src/3rdparty/ym/ym2610.cpp"
        "${NEOCD_DIR}/src/3rdparty/z80/z80.cpp"
        "${NEOCD_DIR}/src/3rdparty/z80/z80daisy.cpp"
    )

    set(NEOCD_C_SOURCES
        "${NEOCD_DIR}/src/3rdparty/musashi/m68kcpu.c"
        "${NEOCD_DIR}/src/3rdparty/musashi/m68kops.c"
        "${NEOCD_DIR}/src/3rdparty/musashi/softfloat/softfloat.c"
        "${LIBCHDR_DIR}/src/libchdr_bitstream.c"
        "${LIBCHDR_DIR}/src/libchdr_cdrom.c"
        "${LIBCHDR_DIR}/src/libchdr_chd.c"
        "${LIBCHDR_DIR}/src/libchdr_flac.c"
        "${LIBCHDR_DIR}/src/libchdr_huffman.c"
        "${LIBRETRO_COMMON_DIR}/compat/compat_posix_string.c"
        "${LIBRETRO_COMMON_DIR}/compat/compat_strcasestr.c"
        "${LIBRETRO_COMMON_DIR}/compat/compat_strl.c"
        "${LIBRETRO_COMMON_DIR}/compat/fopen_utf8.c"
        "${LIBRETRO_COMMON_DIR}/encodings/encoding_crc32.c"
        "${LIBRETRO_COMMON_DIR}/encodings/encoding_utf.c"
        "${LIBRETRO_COMMON_DIR}/file/file_path_io.c"
        "${LIBRETRO_COMMON_DIR}/file/file_path.c"
        "${LIBRETRO_COMMON_DIR}/file/retro_dirent.c"
        "${LIBRETRO_COMMON_DIR}/lists/dir_list.c"
        "${LIBRETRO_COMMON_DIR}/lists/file_list.c"
        "${LIBRETRO_COMMON_DIR}/lists/string_list.c"
        "${LIBRETRO_COMMON_DIR}/memmap/memalign.c"
        "${LIBRETRO_COMMON_DIR}/streams/file_stream_transforms.c"
        "${LIBRETRO_COMMON_DIR}/streams/file_stream.c"
        "${LIBRETRO_COMMON_DIR}/streams/trans_stream_pipe.c"
        "${LIBRETRO_COMMON_DIR}/streams/trans_stream_zlib.c"
        "${LIBRETRO_COMMON_DIR}/streams/trans_stream.c"
        "${LIBRETRO_COMMON_DIR}/string/stdstring.c"
        "${LIBRETRO_COMMON_DIR}/time/rtime.c"
        "${LIBRETRO_COMMON_DIR}/vfs/vfs_implementation.c"
        "${LIBRETRO_COMMON_DIR}/cdrom/cdrom.c"
        "${LIBRETRO_COMMON_DIR}/vfs/vfs_implementation_cdrom.c"
        "${LZMA_DIR}/src/Alloc.c"
        "${LZMA_DIR}/src/Bra.c"
        "${LZMA_DIR}/src/Bra86.c"
        "${LZMA_DIR}/src/BraIA64.c"
        "${LZMA_DIR}/src/CpuArch.c"
        "${LZMA_DIR}/src/Delta.c"
        "${LZMA_DIR}/src/LzFind.c"
        "${LZMA_DIR}/src/Lzma86Dec.c"
        "${LZMA_DIR}/src/LzmaDec.c"
        "${LZMA_DIR}/src/LzmaEnc.c"
        "${LZMA_DIR}/src/Sort.c"
        "${MINIZIP_DIR}/ioapi.c"
        "${MINIZIP_DIR}/unzip.c"
    )

    add_library(neocd_libretro SHARED ${NEOCD_CXX_SOURCES} ${NEOCD_C_SOURCES})
    set_target_properties(neocd_libretro PROPERTIES
        OUTPUT_NAME "neocd_libretro"
        WINDOWS_EXPORT_ALL_SYMBOLS ON
    )

    target_include_directories(neocd_libretro PRIVATE
        "${NEOCD_DIR}/src"
        "${LIBRETRO_COMMON_DIR}/include"
        "${NEOCD_DEPS_DIR}"
        "${LIBCHDR_DIR}/include"
        "${LZMA_DIR}/include"
        "${MINIZIP_DIR}"
    )

    target_compile_definitions(neocd_libretro PRIVATE
        __LIBRETRO__
        HAVE_CDROM
        HAVE_ZLIB
        HAVE_FLAC
        USE_LIBRETRO_VFS
        Z7_ST
        ZSTD_DISABLE_ASM
        _CRT_SECURE_NO_WARNINGS
        _CRT_NONSTDC_NO_DEPRECATE
        WIN32_LEAN_AND_MEAN
        NOMINMAX
        _WIN32_WINNT=0x0601
    )

    target_link_libraries(neocd_libretro PRIVATE
        ZLIB::ZLIB
        $<IF:$<TARGET_EXISTS:zstd::libzstd_shared>,zstd::libzstd_shared,zstd::libzstd_static>
        Ogg::ogg
        Vorbis::vorbisfile
        Vorbis::vorbis
    )

    if(MSVC)
        set_source_files_properties(${NEOCD_CXX_SOURCES} PROPERTIES COMPILE_OPTIONS "/FIcstdlib")
        target_compile_options(neocd_libretro PRIVATE
            /W3
            /wd4244
            /wd4267
            /wd4996
        )
    endif()
endif()

if(QNEOGEO_BUILD_TOOLS)
    add_executable(neocd_bios_probe tools/neocd_bios_probe.cpp)
    target_compile_features(neocd_bios_probe PRIVATE cxx_std_17)
    target_include_directories(neocd_bios_probe PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/neocd_libretro/deps/libretro-common/include"
    )
endif()

if(QNEOGEO_BUILD_QT_APP)
    set(CMAKE_AUTOMOC ON)
    set(CMAKE_AUTOUIC ON)
    set(CMAKE_AUTORCC ON)

    find_package(Qt6 REQUIRED COMPONENTS Widgets OpenGL OpenGLWidgets)
    find_package(RtAudio CONFIG REQUIRED)
    find_package(SampleRate CONFIG REQUIRED)

    set(QNEOGEO_SHADER_FILES
        shaders/6xbrz.glsl
        shaders/README.md
        shaders/frame.frag
        shaders/frame.vert
        shaders/scanline-fract.glsl
        shaders/xbrz-freescale.glsl
        shaders/zfast_crt.glsl
        shaders/zfast_lcd.glsl
    )
    set_source_files_properties(${QNEOGEO_SHADER_FILES} PROPERTIES HEADER_FILE_ONLY TRUE)
    source_group("Shaders" FILES ${QNEOGEO_SHADER_FILES})

    if(QNEOGEO_FETCH_FBNEO_CORE)
        set(FBNEO_CORE_URL "https://buildbot.libretro.com/nightly/windows/x86_64/latest/fbneo_libretro.dll.zip")
        set(FBNEO_CORE_ZIP "${CMAKE_CURRENT_SOURCE_DIR}/downloads/fbneo_libretro.dll.zip")
        set(FBNEO_CORE_EXTRACT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/downloads/fbneo_libretro")
        set(FBNEO_CORE_DLL "${FBNEO_CORE_EXTRACT_DIR}/fbneo_libretro.dll")

        if(NOT EXISTS "${FBNEO_CORE_DLL}")
            file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/downloads")
            message(STATUS "Downloading fbneo_libretro.dll from ${FBNEO_CORE_URL}")
            file(DOWNLOAD
                "${FBNEO_CORE_URL}"
                "${FBNEO_CORE_ZIP}"
                SHOW_PROGRESS
                STATUS FBNEO_DOWNLOAD_STATUS
            )
            list(GET FBNEO_DOWNLOAD_STATUS 0 FBNEO_DOWNLOAD_RESULT)
            list(GET FBNEO_DOWNLOAD_STATUS 1 FBNEO_DOWNLOAD_MESSAGE)
            if(NOT FBNEO_DOWNLOAD_RESULT EQUAL 0)
                message(FATAL_ERROR "Could not download fbneo_libretro.dll: ${FBNEO_DOWNLOAD_MESSAGE}")
            endif()

            file(REMOVE_RECURSE "${FBNEO_CORE_EXTRACT_DIR}")
            file(MAKE_DIRECTORY "${FBNEO_CORE_EXTRACT_DIR}")
            file(ARCHIVE_EXTRACT INPUT "${FBNEO_CORE_ZIP}" DESTINATION "${FBNEO_CORE_EXTRACT_DIR}")
        endif()
    endif()

    qt_add_executable(qneogeo
        emulatorview.cpp
        emulatorview.h
        fbneolibretrocore.cpp
        fbneolibretrocore.h
        iemulatorcore.cpp
        iemulatorcore.h
        inputmappingwidget.cpp
        inputmappingwidget.h
        kof98hitboxoverlay.cpp
        kof98hitboxoverlay.h
        libretrocore.cpp
        libretrocore.h
        main.cpp
        mainwindow.cpp
        mainwindow.h
        mainwindow.ui
        memorysearchdialog.cpp
        memorysearchdialog.h
        neocdlibretrocore.cpp
        neocdlibretrocore.h
        qneogeo.ico
        qneogeo.qrc
        qneogeo.rc
        wasapiaudio.cpp
        wasapiaudio.h
        ${QNEOGEO_SHADER_FILES}
    )

    set_target_properties(qneogeo PROPERTIES
        WIN32_EXECUTABLE "$<CONFIG:Release>"
    )

    target_include_directories(qneogeo PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/neocd_libretro/deps/libretro-common/include"
    )

    target_link_libraries(qneogeo PRIVATE Qt6::Widgets Qt6::OpenGL Qt6::OpenGLWidgets RtAudio::rtaudio SampleRate::samplerate xinput)

    if(TARGET neocd_libretro)
        add_dependencies(qneogeo neocd_libretro)
    endif()

    add_custom_command(TARGET qneogeo POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
            "$<TARGET_FILE_DIR:qneogeo>/shaders"
    )

    if(QNEOGEO_FETCH_FBNEO_CORE AND EXISTS "${FBNEO_CORE_DLL}")
        add_custom_command(TARGET qneogeo POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${FBNEO_CORE_DLL}"
                "$<TARGET_FILE_DIR:qneogeo>/fbneo_libretro.dll"
        )
    endif()
endif()
