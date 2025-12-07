#ifndef DIALOGUPDATESAMPLESIMAGEDIRECTORY_H
#define DIALOGUPDATESAMPLESIMAGEDIRECTORY_H

#include <QDialog>
#include <QString>
#include <QFileDialog>
#include <QCloseEvent>

namespace Ui {
class DialogUpdateSamplesImageDirectory;
}

class DialogUpdateSamplesImageDirectory : public QDialog
{
    Q_OBJECT

public:
    explicit DialogUpdateSamplesImageDirectory(QWidget *parent = 0);
    DialogUpdateSamplesImageDirectory(QWidget *parent = 0,QString prefix="",QString directory="");
    ~DialogUpdateSamplesImageDirectory();

private slots:
    void on_pushButtonBrowserDirectory_clicked();
    void on_pushButtonCancel_clicked();
    void on_pushButtonOk_clicked();

private:
    Ui::DialogUpdateSamplesImageDirectory *ui;

signals:
    void updateDirectoryAndPrefix(QString prefix, QString DirectoryPath);
    void windowClosed(void);
protected:
    void closeEvent(QCloseEvent *event);
};

#endif // DIALOGUPDATESAMPLESIMAGEDIRECTORY_H
