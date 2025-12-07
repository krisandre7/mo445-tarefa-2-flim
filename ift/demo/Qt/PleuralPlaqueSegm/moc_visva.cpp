/****************************************************************************
** Meta object code from reading C++ file 'visva.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.5.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "visva.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'visva.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.5.1. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
struct qt_meta_stringdata_Visva_t {
    QByteArrayData data[13];
    char stringdata0[110];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_Visva_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_Visva_t qt_meta_stringdata_Visva = {
    {
QT_MOC_LITERAL(0, 0, 5), // "Visva"
QT_MOC_LITERAL(1, 6, 8), // "OpenFile"
QT_MOC_LITERAL(2, 15, 0), // ""
QT_MOC_LITERAL(3, 16, 8), // "SaveFile"
QT_MOC_LITERAL(4, 25, 7), // "NewFile"
QT_MOC_LITERAL(5, 33, 6), // "ZoomIn"
QT_MOC_LITERAL(6, 40, 7), // "ZoomOut"
QT_MOC_LITERAL(7, 48, 10), // "NormalSize"
QT_MOC_LITERAL(8, 59, 5), // "About"
QT_MOC_LITERAL(9, 65, 18), // "OpenBrightContrWin"
QT_MOC_LITERAL(10, 84, 10), // "Brightness"
QT_MOC_LITERAL(11, 95, 5), // "value"
QT_MOC_LITERAL(12, 101, 8) // "Contrast"

    },
    "Visva\0OpenFile\0\0SaveFile\0NewFile\0"
    "ZoomIn\0ZoomOut\0NormalSize\0About\0"
    "OpenBrightContrWin\0Brightness\0value\0"
    "Contrast"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_Visva[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
      10,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    0,   64,    2, 0x08 /* Private */,
       3,    0,   65,    2, 0x08 /* Private */,
       4,    0,   66,    2, 0x08 /* Private */,
       5,    0,   67,    2, 0x08 /* Private */,
       6,    0,   68,    2, 0x08 /* Private */,
       7,    0,   69,    2, 0x08 /* Private */,
       8,    0,   70,    2, 0x08 /* Private */,
       9,    0,   71,    2, 0x08 /* Private */,
      10,    1,   72,    2, 0x08 /* Private */,
      12,    1,   75,    2, 0x08 /* Private */,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int,   11,
    QMetaType::Void, QMetaType::Double,   11,

       0        // eod
};

void Visva::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Visva *_t = static_cast<Visva *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->OpenFile(); break;
        case 1: _t->SaveFile(); break;
        case 2: _t->NewFile(); break;
        case 3: _t->ZoomIn(); break;
        case 4: _t->ZoomOut(); break;
        case 5: _t->NormalSize(); break;
        case 6: _t->About(); break;
        case 7: _t->OpenBrightContrWin(); break;
        case 8: _t->Brightness((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 9: _t->Contrast((*reinterpret_cast< double(*)>(_a[1]))); break;
        default: ;
        }
    }
}

const QMetaObject Visva::staticMetaObject = {
    { &QMainWindow::staticMetaObject, qt_meta_stringdata_Visva.data,
      qt_meta_data_Visva,  qt_static_metacall, Q_NULLPTR, Q_NULLPTR}
};


const QMetaObject *Visva::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *Visva::qt_metacast(const char *_clname)
{
    if (!_clname) return Q_NULLPTR;
    if (!strcmp(_clname, qt_meta_stringdata_Visva.stringdata0))
        return static_cast<void*>(const_cast< Visva*>(this));
    return QMainWindow::qt_metacast(_clname);
}

int Visva::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QMainWindow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 10)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 10;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 10)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 10;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
