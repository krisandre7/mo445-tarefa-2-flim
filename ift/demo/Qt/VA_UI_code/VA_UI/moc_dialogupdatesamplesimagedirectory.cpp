/****************************************************************************
** Meta object code from reading C++ file 'dialogupdatesamplesimagedirectory.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.7.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "dialogupdatesamplesimagedirectory.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'dialogupdatesamplesimagedirectory.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.7.1. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
struct qt_meta_stringdata_DialogUpdateSamplesImageDirectory_t {
    QByteArrayData data[9];
    char stringdata0[184];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_DialogUpdateSamplesImageDirectory_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_DialogUpdateSamplesImageDirectory_t qt_meta_stringdata_DialogUpdateSamplesImageDirectory = {
    {
QT_MOC_LITERAL(0, 0, 33), // "DialogUpdateSamplesImageDirec..."
QT_MOC_LITERAL(1, 34, 24), // "updateDirectoryAndPrefix"
QT_MOC_LITERAL(2, 59, 0), // ""
QT_MOC_LITERAL(3, 60, 6), // "prefix"
QT_MOC_LITERAL(4, 67, 13), // "DirectoryPath"
QT_MOC_LITERAL(5, 81, 12), // "windowClosed"
QT_MOC_LITERAL(6, 94, 37), // "on_pushButtonBrowserDirectory..."
QT_MOC_LITERAL(7, 132, 27), // "on_pushButtonCancel_clicked"
QT_MOC_LITERAL(8, 160, 23) // "on_pushButtonOk_clicked"

    },
    "DialogUpdateSamplesImageDirectory\0"
    "updateDirectoryAndPrefix\0\0prefix\0"
    "DirectoryPath\0windowClosed\0"
    "on_pushButtonBrowserDirectory_clicked\0"
    "on_pushButtonCancel_clicked\0"
    "on_pushButtonOk_clicked"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_DialogUpdateSamplesImageDirectory[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       5,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       2,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    2,   39,    2, 0x06 /* Public */,
       5,    0,   44,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
       6,    0,   45,    2, 0x08 /* Private */,
       7,    0,   46,    2, 0x08 /* Private */,
       8,    0,   47,    2, 0x08 /* Private */,

 // signals: parameters
    QMetaType::Void, QMetaType::QString, QMetaType::QString,    3,    4,
    QMetaType::Void,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,

       0        // eod
};

void DialogUpdateSamplesImageDirectory::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        DialogUpdateSamplesImageDirectory *_t = static_cast<DialogUpdateSamplesImageDirectory *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->updateDirectoryAndPrefix((*reinterpret_cast< QString(*)>(_a[1])),(*reinterpret_cast< QString(*)>(_a[2]))); break;
        case 1: _t->windowClosed(); break;
        case 2: _t->on_pushButtonBrowserDirectory_clicked(); break;
        case 3: _t->on_pushButtonCancel_clicked(); break;
        case 4: _t->on_pushButtonOk_clicked(); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        void **func = reinterpret_cast<void **>(_a[1]);
        {
            typedef void (DialogUpdateSamplesImageDirectory::*_t)(QString , QString );
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&DialogUpdateSamplesImageDirectory::updateDirectoryAndPrefix)) {
                *result = 0;
                return;
            }
        }
        {
            typedef void (DialogUpdateSamplesImageDirectory::*_t)();
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&DialogUpdateSamplesImageDirectory::windowClosed)) {
                *result = 1;
                return;
            }
        }
    }
}

const QMetaObject DialogUpdateSamplesImageDirectory::staticMetaObject = {
    { &QDialog::staticMetaObject, qt_meta_stringdata_DialogUpdateSamplesImageDirectory.data,
      qt_meta_data_DialogUpdateSamplesImageDirectory,  qt_static_metacall, Q_NULLPTR, Q_NULLPTR}
};


const QMetaObject *DialogUpdateSamplesImageDirectory::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *DialogUpdateSamplesImageDirectory::qt_metacast(const char *_clname)
{
    if (!_clname) return Q_NULLPTR;
    if (!strcmp(_clname, qt_meta_stringdata_DialogUpdateSamplesImageDirectory.stringdata0))
        return static_cast<void*>(const_cast< DialogUpdateSamplesImageDirectory*>(this));
    return QDialog::qt_metacast(_clname);
}

int DialogUpdateSamplesImageDirectory::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QDialog::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 5)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 5;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 5)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 5;
    }
    return _id;
}

// SIGNAL 0
void DialogUpdateSamplesImageDirectory::updateDirectoryAndPrefix(QString _t1, QString _t2)
{
    void *_a[] = { Q_NULLPTR, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void DialogUpdateSamplesImageDirectory::windowClosed()
{
    QMetaObject::activate(this, &staticMetaObject, 1, Q_NULLPTR);
}
QT_END_MOC_NAMESPACE
