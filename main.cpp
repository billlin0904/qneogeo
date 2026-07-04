#include "mainwindow.h"

#include <QApplication>
#include <QIcon>

int main(int argc, char *argv[]) {
    QApplication a(argc, argv);
    a.setWindowIcon(QIcon(QStringLiteral(":/qneogeo.ico")));
    MainWindow w;
    w.show();
    return a.exec();
}
