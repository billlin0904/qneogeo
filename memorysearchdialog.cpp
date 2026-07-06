#include "memorysearchdialog.h"

#include "libretrocore.h"

#include <QAction>
#include <QBrush>
#include <QColor>
#include <QFont>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QLineEdit>
#include <QMenu>
#include <QPushButton>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QVBoxLayout>

#include <algorithm>

MemorySearchDialog::MemorySearchDialog(LibretroCore *core, QWidget *parent)
    : QDialog(parent)
    , core_(core) {
    setWindowTitle(QStringLiteral("Memory View"));
    setAttribute(Qt::WA_DeleteOnClose);
    resize(640, 520);

    const QFont mono_font(QStringLiteral("SF Mono"), 10);

    auto *main_layout = new QVBoxLayout(this);
    main_layout->setContentsMargins(12, 12, 12, 12);
    main_layout->setSpacing(8);

    auto *controls = new QHBoxLayout;
    controls->addWidget(new QLabel(QStringLiteral("Address"), this));

    dump_address_edit_ = new QLineEdit(QStringLiteral("0x100000"), this);
    dump_address_edit_->setFont(mono_font);
    dump_address_edit_->setMaximumWidth(140);
    controls->addWidget(dump_address_edit_);

    dump_refresh_button_ = new QPushButton(QStringLiteral("Refresh"), this);
    controls->addWidget(dump_refresh_button_);
    controls->addStretch(1);
    main_layout->addLayout(controls);

    dump_table_ = new QTableWidget(this);
    dump_table_->setFont(mono_font);
    dump_table_->horizontalHeader()->setFont(mono_font);
    dump_table_->verticalHeader()->setVisible(false);
    dump_table_->setSelectionBehavior(QAbstractItemView::SelectItems);
    dump_table_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    dump_table_->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
    dump_table_->setContextMenuPolicy(Qt::CustomContextMenu);
    main_layout->addWidget(dump_table_, 1);

    dump_status_label_ = new QLabel(QStringLiteral("Right-click table to switch 1 Byte / 2 Bytes LE."), this);
    dump_status_label_->setFont(mono_font);
    main_layout->addWidget(dump_status_label_);

    rebuildMemoryDumpColumns();

    connect(dump_refresh_button_, &QPushButton::clicked, this, &MemorySearchDialog::refreshMemoryDump);
    connect(dump_address_edit_, &QLineEdit::returnPressed, this, &MemorySearchDialog::refreshMemoryDump);
    connect(dump_table_, &QTableWidget::customContextMenuRequested, this, &MemorySearchDialog::showMemoryDumpContextMenu);

    refreshMemoryDump();
}

void MemorySearchDialog::setCore(LibretroCore *core) {
    core_ = core;
    previous_ram_.clear();
    refreshMemoryDump();
}

void MemorySearchDialog::refreshMemoryDump() {
    if (!dump_table_)
        return;

    QByteArray ram;
    if (!readRam(ram)) {
        dump_table_->setRowCount(0);
        if (dump_status_label_)
            dump_status_label_->setText(QStringLiteral("System RAM is unavailable."));
        return;
    }

    if (ram.isEmpty()) {
        dump_table_->setRowCount(0);
        if (dump_status_label_)
            dump_status_label_->setText(QStringLiteral("System RAM is empty."));
        return;
    }

    rebuildMemoryDumpColumns();

    constexpr int row_count = 32;
    constexpr int bytes_per_row = 16;
    const uint32_t base_offset = dumpBaseOffset(ram);
    const bool has_previous_ram = !previous_ram_.isEmpty();
    dump_table_->setRowCount(row_count);

    const QColor changed_background(92, 72, 16);
    const QColor changed_foreground(255, 236, 170);
    auto applyChangeColor = [&](QTableWidgetItem *item, bool changed) {
        if (!item)
            return;

        if (changed) {
            item->setBackground(changed_background);
            item->setForeground(changed_foreground);
        } else {
            item->setBackground(QBrush());
            item->setForeground(QBrush());
        }
    };

    for (int row = 0; row < row_count; ++row) {
        const uint32_t row_offset = base_offset + static_cast<uint32_t>(row * bytes_per_row);
        auto *address_item = dump_table_->item(row, 0);
        if (!address_item) {
            address_item = new QTableWidgetItem;
            dump_table_->setItem(row, 0, address_item);
        }

        address_item->setText(plainHexAddress(0x100000 + row_offset));
        address_item->setData(Qt::UserRole, row_offset);

        if (dump_unit_ == DumpUnit::Byte1) {
            for (int column = 0; column < bytes_per_row; ++column) {
                auto *item = dump_table_->item(row, column + 1);
                if (!item) {
                    item = new QTableWidgetItem;
                    dump_table_->setItem(row, column + 1, item);
                }

                const uint32_t offset = row_offset + static_cast<uint32_t>(column);
                const bool has_value = offset < static_cast<uint32_t>(ram.size());
                const bool changed = has_previous_ram &&
                    offset < static_cast<uint32_t>(previous_ram_.size()) &&
                    has_value &&
                    ram[static_cast<int>(offset)] != previous_ram_[static_cast<int>(offset)];
                item->setText(offset < static_cast<uint32_t>(ram.size())
                                  ? QStringLiteral("%1").arg(static_cast<uint8_t>(ram[static_cast<int>(offset)]), 2, 16, QLatin1Char('0')).toUpper()
                                  : QStringLiteral("--"));
                item->setData(Qt::UserRole, offset);
                applyChangeColor(item, changed);
            }
        } else {
            for (int column = 0; column < 8; ++column) {
                auto *item = dump_table_->item(row, column + 1);
                if (!item) {
                    item = new QTableWidgetItem;
                    dump_table_->setItem(row, column + 1, item);
                }

                const uint32_t offset = row_offset + static_cast<uint32_t>(column * 2);
                const bool has_value = offset + 1 < static_cast<uint32_t>(ram.size());
                const bool changed = has_previous_ram &&
                    offset + 1 < static_cast<uint32_t>(previous_ram_.size()) &&
                    has_value &&
                    (ram[static_cast<int>(offset)] != previous_ram_[static_cast<int>(offset)] ||
                     ram[static_cast<int>(offset + 1)] != previous_ram_[static_cast<int>(offset + 1)]);
                if (offset + 1 < static_cast<uint32_t>(ram.size())) {
                    const uint16_t value = static_cast<uint16_t>((static_cast<uint8_t>(ram[static_cast<int>(offset + 1)]) << 8) |
                                                                 static_cast<uint8_t>(ram[static_cast<int>(offset)]));
                    item->setText(hexWord(value).mid(2));
                } else {
                    item->setText(QStringLiteral("----"));
                }
                item->setData(Qt::UserRole, offset);
                applyChangeColor(item, changed);
            }
        }
    }

    if (dump_status_label_) {
        const QString unit = dump_unit_ == DumpUnit::Byte1 ? QStringLiteral("1 Byte") : QStringLiteral("2 Bytes LE");
        const uint32_t end_offset = std::min<uint32_t>(base_offset + row_count * bytes_per_row - 1,
                                                       static_cast<uint32_t>(ram.size() - 1));
        dump_status_label_->setText(QStringLiteral("Range: %1-%2  Display: %3")
                                        .arg(plainHexAddress(0x100000 + base_offset),
                                             plainHexAddress(0x100000 + end_offset),
                                             unit));
    }

    previous_ram_ = ram;
}

void MemorySearchDialog::rebuildMemoryDumpColumns() {
    if (!dump_table_)
        return;

    const int value_columns = dump_unit_ == DumpUnit::Byte1 ? 16 : 8;
    if (dump_table_->columnCount() == value_columns + 1)
        return;

    QStringList headers;
    headers.push_back(QStringLiteral("Address"));
    for (int column = 0; column < value_columns; ++column) {
        const int byte_offset = dump_unit_ == DumpUnit::Byte1 ? column : column * 2;
        headers.push_back(QStringLiteral("+%1").arg(byte_offset, 2, 16, QLatin1Char('0')).toUpper());
    }

    dump_table_->clear();
    dump_table_->setColumnCount(value_columns + 1);
    dump_table_->setHorizontalHeaderLabels(headers);
}

void MemorySearchDialog::showMemoryDumpContextMenu(const QPoint &position) {
    QMenu menu(this);
    auto *byte_action = menu.addAction(QStringLiteral("Display as 1 Byte"));
    byte_action->setCheckable(true);
    byte_action->setChecked(dump_unit_ == DumpUnit::Byte1);

    auto *word_action = menu.addAction(QStringLiteral("Display as 2 Bytes LE"));
    word_action->setCheckable(true);
    word_action->setChecked(dump_unit_ == DumpUnit::Byte2);

    QAction *selected = menu.exec(dump_table_->viewport()->mapToGlobal(position));
    if (selected == byte_action)
        setDumpUnit(DumpUnit::Byte1);
    else if (selected == word_action)
        setDumpUnit(DumpUnit::Byte2);
}

void MemorySearchDialog::setDumpUnit(DumpUnit unit) {
    if (dump_unit_ == unit)
        return;

    dump_unit_ = unit;
    rebuildMemoryDumpColumns();
    refreshMemoryDump();
}

bool MemorySearchDialog::readRam(QByteArray &ram) const {
    return core_ && core_->readSystemRam(ram);
}

uint32_t MemorySearchDialog::dumpBaseOffset(const QByteArray &ram) const {
    if (ram.isEmpty())
        return 0;

    const uint32_t parsed_address = parseDumpAddress();
    uint32_t offset = parsed_address >= 0x100000 ? parsed_address - 0x100000 : parsed_address;
    offset = std::min<uint32_t>(offset, static_cast<uint32_t>(ram.size() - 1));
    return offset & ~0x0fU;
}

uint32_t MemorySearchDialog::parseDumpAddress() const {
    QString text = dump_address_edit_ ? dump_address_edit_->text().trimmed() : QString();
    if (text.isEmpty())
        return 0x100000;

    if (text.startsWith(QStringLiteral("0x"), Qt::CaseInsensitive))
        text = text.mid(2);

    bool ok = false;
    const uint32_t value = text.toUInt(&ok, 16);
    return ok ? value : 0x100000;
}

QString MemorySearchDialog::hexWord(uint16_t value) {
    return QStringLiteral("0x%1").arg(value, 4, 16, QLatin1Char('0')).toUpper();
}

QString MemorySearchDialog::plainHexAddress(uint32_t address) {
    return QStringLiteral("%1").arg(address, 6, 16, QLatin1Char('0')).toUpper();
}
