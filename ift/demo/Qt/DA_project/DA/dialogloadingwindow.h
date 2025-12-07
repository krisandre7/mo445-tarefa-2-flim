#ifndef DIALOGLOADINGWINDOW_H
#define DIALOGLOADINGWINDOW_H

#include <QDialog>

namespace Ui {
class DialogLoadingWindow;
}

class DialogLoadingWindow : public QDialog
{
    Q_OBJECT

public:
    explicit DialogLoadingWindow(QWidget *parent = 0);
    Ui::DialogLoadingWindow *ui;
    ~DialogLoadingWindow();

private:


private slots:
    void updateProgressBar(int percentage);
};

#endif // DIALOGLOADINGWINDOW_H
