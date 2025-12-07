/****************************************************************************
** Meta object code from reading C++ file 'graphwidget.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.8.0)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "graphwidget.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'graphwidget.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.8.0. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_GraphWidget_t {
    QByteArrayData data[14];
    char stringdata0[218];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_GraphWidget_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_GraphWidget_t qt_meta_stringdata_GraphWidget = {
    {
QT_MOC_LITERAL(0, 0, 11), // "GraphWidget"
QT_MOC_LITERAL(1, 12, 16), // "mouseCoordinates"
QT_MOC_LITERAL(2, 29, 0), // ""
QT_MOC_LITERAL(3, 30, 20), // "mouseMoveCoordinates"
QT_MOC_LITERAL(4, 51, 8), // "QPointF*"
QT_MOC_LITERAL(5, 60, 33), // "currentMousePositionInGraphic..."
QT_MOC_LITERAL(6, 94, 38), // "currentMousePositionInSceneCo..."
QT_MOC_LITERAL(7, 133, 12), // "keepPressing"
QT_MOC_LITERAL(8, 146, 12), // "mousePressed"
QT_MOC_LITERAL(9, 159, 15), // "Qt::MouseButton"
QT_MOC_LITERAL(10, 175, 7), // "buttons"
QT_MOC_LITERAL(11, 183, 19), // "deleteObjectSignals"
QT_MOC_LITERAL(12, 203, 6), // "zoomIn"
QT_MOC_LITERAL(13, 210, 7) // "zoomOut"

    },
    "GraphWidget\0mouseCoordinates\0\0"
    "mouseMoveCoordinates\0QPointF*\0"
    "currentMousePositionInGraphicArea\0"
    "currentMousePositionInSceneCoordinates\0"
    "keepPressing\0mousePressed\0Qt::MouseButton\0"
    "buttons\0deleteObjectSignals\0zoomIn\0"
    "zoomOut"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_GraphWidget[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       6,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       4,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    0,   44,    2, 0x06 /* Public */,
       3,    3,   45,    2, 0x06 /* Public */,
       8,    3,   52,    2, 0x06 /* Public */,
      11,    0,   59,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
      12,    0,   60,    2, 0x0a /* Public */,
      13,    0,   61,    2, 0x0a /* Public */,

 // signals: parameters
    QMetaType::Void,
    QMetaType::Void, 0x80000000 | 4, 0x80000000 | 4, QMetaType::Bool,    5,    6,    7,
    QMetaType::Void, 0x80000000 | 4, 0x80000000 | 4, 0x80000000 | 9,    5,    6,   10,
    QMetaType::Void,

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
        case 0: _t->mouseCoordinates(); break;
        case 1: _t->mouseMoveCoordinates((*reinterpret_cast< QPointF*(*)>(_a[1])),(*reinterpret_cast< QPointF*(*)>(_a[2])),(*reinterpret_cast< bool(*)>(_a[3]))); break;
        case 2: _t->mousePressed((*reinterpret_cast< QPointF*(*)>(_a[1])),(*reinterpret_cast< QPointF*(*)>(_a[2])),(*reinterpret_cast< Qt::MouseButton(*)>(_a[3]))); break;
        case 3: _t->deleteObjectSignals(); break;
        case 4: _t->zoomIn(); break;
        case 5: _t->zoomOut(); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        void **func = reinterpret_cast<void **>(_a[1]);
        {
            typedef void (GraphWidget::*_t)();
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&GraphWidget::mouseCoordinates)) {
                *result = 0;
                return;
            }
        }
        {
            typedef void (GraphWidget::*_t)(QPointF * , QPointF * , bool );
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&GraphWidget::mouseMoveCoordinates)) {
                *result = 1;
                return;
            }
        }
        {
            typedef void (GraphWidget::*_t)(QPointF * , QPointF * , Qt::MouseButton );
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&GraphWidget::mousePressed)) {
                *result = 2;
                return;
            }
        }
        {
            typedef void (GraphWidget::*_t)();
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&GraphWidget::deleteObjectSignals)) {
                *result = 3;
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
void GraphWidget::mouseCoordinates()
{
    QMetaObject::activate(this, &staticMetaObject, 0, Q_NULLPTR);
}

// SIGNAL 1
void GraphWidget::mouseMoveCoordinates(QPointF * _t1, QPointF * _t2, bool _t3)
{
    void *_a[] = { Q_NULLPTR, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)), const_cast<void*>(reinterpret_cast<const void*>(&_t3)) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}

// SIGNAL 2
void GraphWidget::mousePressed(QPointF * _t1, QPointF * _t2, Qt::MouseButton _t3)
{
    void *_a[] = { Q_NULLPTR, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)), const_cast<void*>(reinterpret_cast<const void*>(&_t3)) };
    QMetaObject::activate(this, &staticMetaObject, 2, _a);
}

// SIGNAL 3
void GraphWidget::deleteObjectSignals()
{
    QMetaObject::activate(this, &staticMetaObject, 3, Q_NULLPTR);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
