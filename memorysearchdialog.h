#pragma once

#include <QByteArray>
#include <QDialog>

#include <cstdint>

class QLabel;
class QLineEdit;
class QPushButton;
class QTableWidget;
class LibretroCore;

class MemorySearchDialog final : public QDialog {
    Q_OBJECT

public:
    explicit MemorySearchDialog(LibretroCore *core, QWidget *parent = nullptr);

    void setCore(LibretroCore *core);

private:
    enum class DumpUnit {
        Byte1,
        Byte2
    };

    void refreshMemoryDump();
    void rebuildMemoryDumpColumns();
    void showMemoryDumpContextMenu(const QPoint &position);
    void setDumpUnit(DumpUnit unit);
    bool readRam(QByteArray &ram) const;
    uint32_t dumpBaseOffset(const QByteArray &ram) const;
    uint32_t parseDumpAddress() const;
    static QString hexWord(uint16_t value);
    static QString plainHexAddress(uint32_t address);

    LibretroCore *core_ = nullptr;
    QByteArray previous_ram_;
    QLabel *dump_status_label_ = nullptr;
    QPushButton *dump_refresh_button_ = nullptr;
    QLineEdit *dump_address_edit_ = nullptr;
    QTableWidget *dump_table_ = nullptr;
    DumpUnit dump_unit_ = DumpUnit::Byte1;
};
