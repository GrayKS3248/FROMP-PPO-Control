CFLAGS-L+=-I/usr/include/python3.8 -I/usr/include/python3.8
CFLAGS-L+=-Wno-unused-result -Wsign-compare -specs=/usr/share/dpkg/no-pie-compile.specs -fstack-protector -Wformat -Werror=format-security -fwrapv -O2 -Wall -std=c++2a -fopenmp -fPIE -fconcepts
LDFLAGS-L+=-L/usr/lib/python3.8/config-3.8-x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu
LDFLAGS-L+=-lcrypt -lpthread -ldl -lutil -lm -lpython3.8

CFLAGS-W+=-I$(CPYTHONPATH)
CFLAGS-W+=-Wno-unused-result -Wsign-compare -Wformat -Werror=format-security -fwrapv -O2 -Wall -std=c++2a -fopenmp -fconcepts
LDFLAGS-W+=-L$(PYTHONPATH)
LDFLAGS-W+=-lpython39

Config_Handler=src/Config_Handler.hpp src/Config_Handler.cpp
matrix_math=src/matrix_math.hpp src/matrix_math.cpp
python_interface=src/python_interface.hpp src/python_interface.cpp
Speed_Estimator=src/Speed_Estimator.hpp src/Speed_Estimator.cpp
Finite_Difference_Solver=src/Finite_Difference_Solver.hpp src/Finite_Difference_Solver.cpp

all-w: clean-w tuner-w solver-w ae_data-w trainer-w
	
all-l: clean-l tuner-l solver-l ae_data-l trainer-l

tuner-w:
	g++ $(CFLAGS-W) src/Finite_Difference_Solver.h src/Finite_Difference_Solver.cpp src/Tune_Params.cpp -o bin/tuner.exe $(LDFLAGS-W)
solver-w:
	g++ $(CFLAGS-W) $(Config_Handler) $(matrix_math) $(python_interface) $(Speed_Estimator) $(Finite_Difference_Solver) src/solver.cpp -o bin/solver.exe $(LDFLAGS-W)
ae_data-w:
	g++ $(CFLAGS-W) $(Config_Handler) $(matrix_math) $(Finite_Difference_Solver) src/ae_data.cpp -o bin/ae_data.exe $(LDFLAGS-W)
trainer-w:
	g++ $(CFLAGS-W) $(Config_Handler) $(matrix_math) $(python_interface) $(Speed_Estimator) $(Finite_Difference_Solver) src/trainer.cpp -o bin/trainer.exe $(LDFLAGS-W)

tuner-l:
	g++ $(CFLAGS-L) src/Finite_Difference_Solver.h src/Finite_Difference_Solver.cpp src/Tune_Params.cpp -o bin/tuner.so $(LDFLAGS-L)
solver-l:
	g++ $(CFLAGS-L) $(Config_Handler) $(matrix_math) $(python_interface) $(Speed_Estimator) $(Finite_Difference_Solver) src/solver.cpp -o bin/solver.so $(LDFLAGS-L)
ae_data-l:
	g++ $(CFLAGS-L) $(Config_Handler) $(matrix_math) $(Finite_Difference_Solver) src/ae_data.cpp -o bin/ae_data.so $(LDFLAGS-L)
trainer-l:
	g++ $(CFLAGS-L) $(Config_Handler) $(matrix_math) $(python_interface) $(Speed_Estimator) $(Finite_Difference_Solver) src/trainer.cpp -o bin/trainer.so $(LDFLAGS-L)
	
clean-w:
	del /f bin/tuner.exe
	del /f bin/solver.exe
	del /f bin/gen_ae_data.exe
	del /f bin/trainer.exe
	del /f bin/tuner.so
	del /f bin/solver.so
	del /f bin/gen_ae_data.so
	del /f bin/trainer.so
	
clean-l:
	rm -rf bin/tuner.so
	rm -rf bin/solver.so
	rm -rf bin/gen_ae_data.so
	rm -rf bin/trainer.so
	rm -rf bin/tuner.exe
	rm -rf bin/solver.exe
	rm -rf bin/gen_ae_data.exe
	rm -rf bin/trainer.exe