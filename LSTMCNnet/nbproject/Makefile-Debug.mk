#
# Generated Makefile - do not edit!
#
# Edit the Makefile in the project folder instead (../Makefile). Each target
# has a -pre and a -post target defined where you can add customized code.
#
# This makefile implements configuration specific macros and targets.


# Environment
MKDIR=mkdir
CP=cp
GREP=grep
NM=nm
CCADMIN=CCadmin
RANLIB=ranlib
CC=gcc
CCC=g++
CXX=g++
FC=gfortran
AS=as

# Macros
CND_PLATFORM=GNU-Linux
CND_DLIB_EXT=so
CND_CONF=Debug
CND_DISTDIR=dist
CND_BUILDDIR=build

# Include project Makefile
include Makefile

# Object Directory
OBJECTDIR=${CND_BUILDDIR}/${CND_CONF}/${CND_PLATFORM}

# Object Files
OBJECTFILES= \
	${OBJECTDIR}/CNNPredictionModel.o \
	${OBJECTDIR}/CNNet/Activation.o \
	${OBJECTDIR}/CNNet/CNN.o \
	${OBJECTDIR}/CNNet/ConvolutionLayer.o \
	${OBJECTDIR}/CNNet/FCLayer.o \
	${OBJECTDIR}/CNNet/PoolLayer.o \
	${OBJECTDIR}/LSTMCNNFCPredictionModel.o \
	${OBJECTDIR}/LSTMCNNPredictionModel.o \
	${OBJECTDIR}/LSTMPredictionModel.o \
	${OBJECTDIR}/LSTMnet/DataProcessor.o \
	${OBJECTDIR}/LSTMnet/FileProcessor.o \
	${OBJECTDIR}/LSTMnet/LSTMNet.o \
	${OBJECTDIR}/PredictionModel.o


# C Compiler Flags
CFLAGS=

# CC Compiler Flags
CCFLAGS=
CXXFLAGS=

# Fortran Compiler Flags
FFLAGS=

# Assembler Flags
ASFLAGS=

# Link Libraries and Options
LDLIBSOPTIONS=

# Build Targets
.build-conf: ${BUILD_SUBPROJECTS}
	"${MAKE}"  -f nbproject/Makefile-${CND_CONF}.mk ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/libLSTMCNnet.${CND_DLIB_EXT}

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/libLSTMCNnet.${CND_DLIB_EXT}: ${OBJECTFILES}
	${MKDIR} -p ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}
	${LINK.cc} -o ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/libLSTMCNnet.${CND_DLIB_EXT} ${OBJECTFILES} ${LDLIBSOPTIONS} -shared -fPIC

${OBJECTDIR}/CNNPredictionModel.o: CNNPredictionModel.cpp
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.cc) -g -ICNNet/Eigen -std=c++14 -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/CNNPredictionModel.o CNNPredictionModel.cpp

${OBJECTDIR}/CNNet/Activation.o: CNNet/Activation.cpp
	${MKDIR} -p ${OBJECTDIR}/CNNet
	${RM} "$@.d"
	$(COMPILE.cc) -g -ICNNet/Eigen -std=c++14 -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/CNNet/Activation.o CNNet/Activation.cpp

${OBJECTDIR}/CNNet/CNN.o: CNNet/CNN.cpp
	${MKDIR} -p ${OBJECTDIR}/CNNet
	${RM} "$@.d"
	$(COMPILE.cc) -g -ICNNet/Eigen -std=c++14 -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/CNNet/CNN.o CNNet/CNN.cpp

${OBJECTDIR}/CNNet/ConvolutionLayer.o: CNNet/ConvolutionLayer.cpp
	${MKDIR} -p ${OBJECTDIR}/CNNet
	${RM} "$@.d"
	$(COMPILE.cc) -g -ICNNet/Eigen -std=c++14 -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/CNNet/ConvolutionLayer.o CNNet/ConvolutionLayer.cpp

${OBJECTDIR}/CNNet/FCLayer.o: CNNet/FCLayer.cpp
	${MKDIR} -p ${OBJECTDIR}/CNNet
	${RM} "$@.d"
	$(COMPILE.cc) -g -ICNNet/Eigen -std=c++14 -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/CNNet/FCLayer.o CNNet/FCLayer.cpp

${OBJECTDIR}/CNNet/PoolLayer.o: CNNet/PoolLayer.cpp
	${MKDIR} -p ${OBJECTDIR}/CNNet
	${RM} "$@.d"
	$(COMPILE.cc) -g -ICNNet/Eigen -std=c++14 -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/CNNet/PoolLayer.o CNNet/PoolLayer.cpp

${OBJECTDIR}/LSTMCNNFCPredictionModel.o: LSTMCNNFCPredictionModel.cpp
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.cc) -g -ICNNet/Eigen -std=c++14 -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/LSTMCNNFCPredictionModel.o LSTMCNNFCPredictionModel.cpp

${OBJECTDIR}/LSTMCNNPredictionModel.o: LSTMCNNPredictionModel.cpp
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.cc) -g -ICNNet/Eigen -std=c++14 -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/LSTMCNNPredictionModel.o LSTMCNNPredictionModel.cpp

${OBJECTDIR}/LSTMPredictionModel.o: LSTMPredictionModel.cpp
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.cc) -g -ICNNet/Eigen -std=c++14 -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/LSTMPredictionModel.o LSTMPredictionModel.cpp

${OBJECTDIR}/LSTMnet/DataProcessor.o: LSTMnet/DataProcessor.cpp
	${MKDIR} -p ${OBJECTDIR}/LSTMnet
	${RM} "$@.d"
	$(COMPILE.cc) -g -ICNNet/Eigen -std=c++14 -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/LSTMnet/DataProcessor.o LSTMnet/DataProcessor.cpp

${OBJECTDIR}/LSTMnet/FileProcessor.o: LSTMnet/FileProcessor.cpp
	${MKDIR} -p ${OBJECTDIR}/LSTMnet
	${RM} "$@.d"
	$(COMPILE.cc) -g -ICNNet/Eigen -std=c++14 -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/LSTMnet/FileProcessor.o LSTMnet/FileProcessor.cpp

${OBJECTDIR}/LSTMnet/LSTMNet.o: LSTMnet/LSTMNet.cpp
	${MKDIR} -p ${OBJECTDIR}/LSTMnet
	${RM} "$@.d"
	$(COMPILE.cc) -g -ICNNet/Eigen -std=c++14 -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/LSTMnet/LSTMNet.o LSTMnet/LSTMNet.cpp

${OBJECTDIR}/PredictionModel.o: PredictionModel.cpp
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.cc) -g -ICNNet/Eigen -std=c++14 -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/PredictionModel.o PredictionModel.cpp

# Subprojects
.build-subprojects:

# Clean Targets
.clean-conf: ${CLEAN_SUBPROJECTS}
	${RM} -r ${CND_BUILDDIR}/${CND_CONF}

# Subprojects
.clean-subprojects:

# Enable dependency checking
.dep.inc: .depcheck-impl

include .dep.inc
