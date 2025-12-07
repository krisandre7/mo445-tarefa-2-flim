#include <QString>
#include <iftDataSet.h>
#include <QFuture>
#include <QtConcurrent>
#include <datasetutils.h>

DataSetUtils* DataSetUtils::sInstance = new DataSetUtils(); // or NULL, or nullptr in c++11
std::mutex DataSetUtils::m;

void DataSetUtils::loadDataSetThread() {
   printf("opening dataset ...\n");
   dataset = iftReadDataSet(filepath.toLatin1().data());
   printf("Done loading!\n");
   m.unlock();
}

void DataSetUtils::load(QString filepath)
{
    m.lock();
    this->filepath = filepath;
    watcher = new QFutureWatcher<void>(this);
    connect(watcher, &QFutureWatcher<void>::finished, this, &DataSetUtils::threadFinished);

    QFuture<void> future = QtConcurrent::run([=](){this->loadDataSetThread();});
    watcher->setFuture(future);
}

iftDataSet *DataSetUtils::getDataset()
{
    return dataset;
}

void DataSetUtils::threadFinished()
{
    emit loadFinished();
}
