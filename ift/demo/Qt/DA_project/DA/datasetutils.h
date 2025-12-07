#ifndef DATASETUTILS_H
#define DATASETUTILS_H

#include <QFutureWatcher>
#include <QString>
#include <iftDataSet.h>
#include <mutex>

class DataSetUtils : public QObject
{
    Q_OBJECT
private:
    static DataSetUtils *sInstance;
    DataSetUtils() {}
    iftDataSet* dataset;
    QString filepath;
    QFutureWatcher<void>* watcher;
    void loadDataSetThread();
    static std::mutex m;
public:
    static DataSetUtils* instance()
    {
        return sInstance;
    }

    void load(QString filepath);
    iftDataSet* getDataset();

public slots:
    void threadFinished();

signals:
    void loadFinished();
};

#endif // DATASETUTILS_H
