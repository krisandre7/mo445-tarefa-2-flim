#ifndef MAINWINDOW_H
#define MAINWINDOW_H

extern "C" {
#include "ift.h"
}

#include <QMainWindow>
#include <QGraphicsScene>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override;

private slots:
    void loadImage();
    bool saveCanvas();

    void setReconst(bool value);
    void setThreshold(int position);

    bool drawToCanvas();

    void resizeEvent(QResizeEvent *) override;

private:
    void createSignals();

    bool maybeSave();
    bool openImage(const QString &file_name);

    iftImage *createSkel();
	QImage getSkel();
    bool drawMSSkel(const iftImage *bin_img);
	QImage getReconst();

    float getSliderPosition(int position);

private:
    Ui::MainWindow *m_ui;

    bool m_loaded;
    bool m_reconst;
    float m_threshold;

    QGraphicsScene *m_thold_scene;
    QGraphicsScene *m_msskel_scene;

    QImage m_image;
	iftFImage *m_msskel;
    iftImage *m_dist;
};

#endif // MAINWINDOW_H
