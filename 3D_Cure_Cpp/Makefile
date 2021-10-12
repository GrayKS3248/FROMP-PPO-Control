CFLAGS-L+=-I/usr/include/python3.8 -I/usr/include/python3.8
CFLAGS-L+=-Wno-unused-result -Wsign-compare -specs=/usr/share/dpkg/no-pie-compile.specs -fstack-protector -Wformat -Werror=format-security -fwrapv -O2 -Wall -std=c++2a -fopenmp -fPIE -fconcepts
LDFLAGS-L+=-L/usr/lib/python3.8/config-3.8-x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu
LDFLAGS-L+=-lcrypt -lpthread -ldl -lutil -lm -lpython3.8

CFLAGS-W+=-I$(CPYTHONPATH)
CFLAGS-W+=-Wno-unused-result -Wsign-compare -Wformat -Werror=format-security -fwrapv -O2 -Wall -std=c++2a -fopenmp -fconcepts
LDFLAGS-W+=-L$(PYTHONPATH)
LDFLAGS-W+=-lpython39

all-w: tuner-w solver-w autoencoder-w estimator-w trainer-w
	
all-l: tuner-l solver-l autoencoder-l estimator-l trainer-l

tuner-w:
	g++ $(CFLAGS-W) src/Finite_Difference_Solver.h src/Finite_Difference_Solver.cpp src/Tune_Params.cpp -o bin/tuner.exe $(LDFLAGS-W)
solver-w:
	g++ $(CFLAGS-W) src/Finite_Difference_Solver.h src/Finite_Difference_Solver.cpp src/Run_Solver.cpp -o bin/solver.exe $(LDFLAGS-W)
autoencoder-w:
	g++ $(CFLAGS-W) src/Finite_Difference_Solver.h src/Finite_Difference_Solver.cpp src/Get_AE_Data.cpp -o bin/gen_ae_data.exe $(LDFLAGS-W)
estimator-w:
	g++ $(CFLAGS-W) src/Finite_Difference_Solver.h src/Finite_Difference_Solver.cpp src/Get_Estimator_Data.cpp -o bin/gen_est_data.exe $(LDFLAGS-W)
trainer-w:
	g++ $(CFLAGS-W) src/Finite_Difference_Solver.h src/Finite_Difference_Solver.cpp src/Train_Agent.cpp -o bin/trainer.exe $(LDFLAGS-W)

tuner-l:
	g++ $(CFLAGS-L) src/Finite_Difference_Solver.h src/Finite_Difference_Solver.cpp src/Tune_Params.cpp -o bin/tuner.so $(LDFLAGS-L)
solver-l:
	g++ $(CFLAGS-L) src/Finite_Difference_Solver.h src/Finite_Difference_Solver.cpp src/Run_Solver.cpp -o bin/solver.so $(LDFLAGS-L)
autoencoder-l:
	g++ $(CFLAGS-L) src/Finite_Difference_Solver.h src/Finite_Difference_Solver.cpp src/Get_AE_Data.cpp -o bin/gen_ae_data.so $(LDFLAGS-L)
estimator-l:
	g++ $(CFLAGS-L) src/Finite_Difference_Solver.h src/Finite_Difference_Solver.cpp src/Get_Estimator_Data.cpp -o bin/gen_est_data.so $(LDFLAGS-L)
trainer-l:
	g++ $(CFLAGS-L) src/Finite_Difference_Solver.h src/Finite_Difference_Solver.cpp src/Train_Agent.cpp -o bin/trainer.so $(LDFLAGS-L)
	
clean-w:
	del /f bin/tuner.exe
	del /f bin/solver.exe
	del /f bin/gen_ae_data.exe
	del /f bin/gen_est_data.exe
	del /f bin/trainer.exe
	del /f bin/tuner.so
	del /f bin/solver.so
	del /f bin/gen_ae_data.so
	del /f bin/gen_est_data.so
	del /f bin/trainer.so
	
clean-l:
	rm -rf bin/tuner.so
	rm -rf bin/solver.so
	rm -rf bin/gen_ae_data.so
	rm -rf bin/gen_est_data.so
	rm -rf bin/trainer.so
	rm -rf bin/tuner.exe
	rm -rf bin/solver.exe
	rm -rf bin/gen_ae_data.exe
	rm -rf bin/gen_est_data.exe
	rm -rf bin/trainer.exe