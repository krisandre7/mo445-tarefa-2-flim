/****************************************************************************
** Meta object code from reading C++ file 'brightcontrwin.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.5.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "brightcontrwin.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'brightcontrwin.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.5.1. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
struct qt_meta_stringdata_BrightContrWin_t {
    QByteArrayData data[9];
    char stringdata0[118];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_BrightContrWin_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_BrightContrWin_t qt_meta_stringdata_BrightContrWin = {
    {
QT_MOC_LITERAL(0, 0, 14), // "BrightContrWin"
QT_MOC_LITERAL(1, 15, 16), // "changeBrightness"
QT_MOC_LITERAL(2, 32, 0), // ""
QT_MOC_LITERAL(3, 33, 5), // "value"
QT_MOC_LITERAL(4, 39, 14), // "changeContrast"
QT_MOC_LITERAL(5, 54, 16), // "callSignalBright"
QT_MOC_LITERAL(6, 71, 18), // "callSignalContrast"
QT_MOC_LITERAL(7, 90, 14), // "stepBrightness"
QT_MOC_LITERAL(8, 105, 12) // "stepContrast"

    },
    "BrightContrWin\0changeBrightness\0\0value\0"
    "changeContrast\0callSignalBright\0"
    "callSignalContrast\0stepBrightness\0"
    "stepContrast"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_BrightContrWin[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       6,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       2,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    1,   44,    2, 0x06 /* Public */,
       4,    1,   47,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
       5,    0,   50,    2, 0x0a /* Public */,
       6,    0,   51,    2, 0x0a /* Public */,
       7,    0,   52,    2, 0x0a /* Public */,
       8,    0,   53,    2, 0x0a /* Public */,

 // signals: parameters
    QMetaType::Void, QMetaType::Int,    3,
    QMetaType::Void, QMetaType::Double,    3,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,

       0        // eod
};

void BrightContrWin::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        BrightContrWin *_t = static_cast<BrightContrWin *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->changeBrightness((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 1: _t->changeContrast((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 2: _t->callSignalBright(); break;
        case 3: _t->callSignalContrast(); break;
        case 4: _t->stepBrightness(); break;
        case 5: _t->stepContrast(); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        void **func = reinterpret_cast<void **>(_a[1]);
        {
            typedef void (BrightContrWin::*_t)(int );
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&BrightContrWin::changeBrightness)) {
                *result = 0;
            }
        }
        {
            typedef void (BrightContrWin::*_t)(double );
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&BrightContrWin::changeContrast)) {
                *result = 1;
            }
        }
    }
}

const QMetaObject BrightContrWin::staticMetaObject = {
    { &QDialog::staticMetaObject, qt_meta_stringdata_BrightContrWin.data,
      qt_meta_data_BrightContrWin,  qt_static_metacall, Q_NULLPTR, Q_NULLPTR}
};


const QMetaObject *BrightContrWin::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *BrightContrWin::qt_metacast(const char *_clname)
{
    if (!_clname) return Q_NULLPTR;
    if (!strcmp(_clname, qt_meta_stringdata_BrightContrWin.stringdata0))
        return static_cast<void*>(const_cast< BrightContrWin*>(this));
    return QDialog::qt_metacast(_clname);
}

int BrightContrWin::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QDialog::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 6)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 6;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 6)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 6;
    }
    return _id;
}

// SIGNAL 0
void BrightContrWin::changeBrightness(int _t1)
{
    void *_a[] = { Q_NULLPTR, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void BrightContrWin::changeContrast(double _t1)
{
    void *_a[] = { Q_NULLPTR, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}
QT_END_MOC_NAMESPACE
