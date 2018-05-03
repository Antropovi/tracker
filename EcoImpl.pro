TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp \
    eco.cpp \
    sample.cpp \
    cn_data.cpp \
    ffttools.cpp \
    scoresoptimizer.cpp \
    trainer.cpp \
    ecofeatures.cpp

INCLUDEPATH += /usr/local/include/opencv
LIBS += -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_features2d -lopencv_imgproc -lopencv_videoio -lopencv_calib3d -lopencv_tracking

HEADERS += \
    eco.h \
    params.h \
    sample.h \
    ffttools.h \
    scoresoptimizer.h \
    trainer.h \
    recttools.h \
    ecofeatures.h
