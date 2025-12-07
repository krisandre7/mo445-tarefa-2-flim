#include <QApplication>
#include <ift.h>
#include "mainwindow.h"

int main(int argc, char *argv[])
{
  int nScales = (argc >= 3) ? atol(argv[2]) : -1;
  if (nScales <= 0 || argc < 3 + nScales) {
    printf("Usage: %s <orig_img> <n_scales> (<coarsest_superpixels> ... <finest_superpixels>) [groundtruth]\n", argv[0]);
    printf("  Controls:\n");
    printf("    Mouse left/right click to add foreground/background labels\n");
    printf("    Up/Down to change scale\n");
    printf("    0~9 to change segmentation method\n");
    printf("    Z to rollback from last marker addition\n");
    printf("    Q to cycle through segmentation visualization modes\n");
    printf("    W to toggle superpixel hierarchy visualization\n");
    printf("    E to toggle markers visualization\n");
    return -1;
  }

  char *gtPath = (argc > 3 + nScales) ? argv[3 + nScales] : nullptr;

  QApplication app(argc, argv);
  MainWindow window(argv[1], &(argv[3]), nScales, gtPath);
  window.show();
  return app.exec();
}

