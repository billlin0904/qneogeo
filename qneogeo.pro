QT       += core gui opengl openglwidgets

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    emulatorview.cpp \
    inputmappingwidget.cpp \
    libretrocore.cpp \
    main.cpp \
    mainwindow.cpp \
    wasapiaudio.cpp

HEADERS += \
    emulatorview.h \
    inputmappingwidget.h \
    libretrocore.h \
    mainwindow.h \
    memory.h \
    wasapiaudio.h

INCLUDEPATH += \
    thirdparty/neocd_libretro/deps/libretro-common/include \
    F:/vcpkg/vcpkg/installed/x64-windows/include

win32:LIBS += -LF:/vcpkg/vcpkg/installed/x64-windows/lib -lsamplerate -lole32 -luuid -lxinput

FORMS += \
    mainwindow.ui

TRANSLATIONS += \
    zh_TW.ts

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
