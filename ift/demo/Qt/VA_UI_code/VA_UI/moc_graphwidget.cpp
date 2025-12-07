/****************************************************************************
** Meta object code from reading C++ file 'graphwidget.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.7.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "graphwidget.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'graphwidget.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.7.1. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
struct qt_meta_stringdata_GraphWidget_t {
    QByteArrayData data[12];
    char stringdata0[199];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_GraphWidget_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_GraphWidget_t qt_meta_stringdata_GraphWidget = {
    {
QT_MOC_LITERAL(0, 0, 11), // "GraphWidget"
QT_MOC_LITERAL(1, 12, 20), // "SelectedOptionSginal"
QT_MOC_LITERAL(2, 33, 0), // ""
QT_MOC_LITERAL(3, 34, 14), // "DrawingManager"
QT_MOC_LITERAL(4, 49, 14), // "drawingManager"
QT_MOC_LITERAL(5, 64, 16), // "mouseCoordinates"
QT_MOC_LITERAL(6, 81, 20), // "mouseMoveCoordinates"
QT_MOC_LITERAL(7, 102, 8), // "QPointF*"
QT_MOC_LITERAL(8, 111, 33), // "currentMousePositionInGraphic..."
QT_MOC_LITERAL(9, 145, 38), // "currentMousePositionInSceneCo..."
QT_MOC_LITERAL(10, 184, 6), // "zoomIn"
QT_MOC_LITERAL(11, 191, 7) // "zoomOut"

    },
    "GraphWidget\0SelectedOptionSginal\0\0"
    "DrawingManager\0drawingManager\0"
    "mouseCoordinates\0mouseMoveCoordinates\0"
    "QPointF*\0currentMousePositionInGraphicArea\0"
    "currentMousePositionInSceneCoordinates\0"
    "zoomIn\0zoomOut"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_GraphWidget[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       5,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       3,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    1,   39,    2, 0x06 /* Public */,
       5,    0,   42,    2, 0x06 /* Public */,
       6,    2,   43,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
      10,    0,   48,    2, 0x0a /* Public */,
      11,    0,   49,    2, 0x0a /* Public */,

 // signals: parameters
    QMetaType::Void, 0x80000000 | 3,    4,
    QMetaType::Void,
    QMetaType::Void, 0x80000000 | 7, 0x80000000 | 7,    8,    9,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void,

       0        // eod
};

void GraphWidget::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        GraphWidget *_t = static_cast<GraphWidget *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->SelectedOptionSginal((*reinterpret_cast< DrawingManager(*)>(_a[1]))); break;
        case 1: _t->mouseCoordinates(); break;
        case 2: _t->mouseMoveCoordinates((*reinterpret_cast< QPointF*(*)>(_a[1])),(*reinterpret_cast< QPointF*(*)>(_a[2]))); break;
        case 3: _t->zoomIn(); break;
        case 4: _t->zoomOut(); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        void **func = reinterpret_cast<void **>(_a[1]);
        {
            typedef void (GraphWidget::*_t)(DrawingManager );
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&GraphWidget::SelectedOptionSginal)) {
                *result = 0;
                return;
            }
        }
        {
            typedef void (GraphWidget::*_t)();
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&GraphWidget::mouseCoordinates)) {
                *result = 1;
                return;
            }
        }
        {
            typedef void (GraphWidget::*_t)(QPointF * , QPointF * );
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&GraphWidget::mouseMoveCoordinates)) {
                *result = 2;
                return;
            }
        }
    }
}

const QMetaObject GraphWidget::staticMetaObject = {
    { &QGraphicsView::staticMetaObject, qt_meta_stringdata_GraphWidget.data,
      qt_meta_data_GraphWidget,  qt_static_metacall, Q_NULLPTR, Q_NULLPTR}
};


const QMetaObject *GraphWidget::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *GraphWidget::qt_metacast(const char *_clname)
{
    if (!_clname) return Q_NULLPTR;
    if (!strcmp(_clname, qt_meta_stringdata_GraphWidget.stringdata0))
        return static_cast<void*>(const_cast< GraphWidget*>(this));
    return QGraphicsView::qt_metacast(_clname);
}

int GraphWidget::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QGraphicsView::qt_metacall(_c, _id, _a);
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
void GraphWidget::SelectedOptionSginal(DrawingManager _t1)
{
    void *_a[] = { Q_NULLPTR, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void GraphWidget::mouseCoordinates()
{
    QMetaObject::activate(this, &staticMetaObject, 1, Q_NULLPTR);
}

// SIGNAL 2
void GraphWidget::mouseMoveCoordinates(QPointF * _t1, QPointF * _t2)
{
    void *_a[] = { Q_NULLPTR, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 2, _a);
}
QT_END_MOC_NAMESPACE
