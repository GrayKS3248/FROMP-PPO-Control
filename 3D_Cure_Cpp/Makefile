CFLAGS-L+=-I/usr/include/python3.8 -I/usr/include/python3.8
CFLAGS-L+=-Wno-unused-result -Wsign-compare -specs=/usr/share/dpkg/no-pie-compile.specs -fstack-protector -Wformat -Werror=format-security -fwrapv -O2 -Wall -std=c++2a -fopenmp -fPIE -fconcepts
LDFLAGS-L+=-L/usr/lib/python3.8/config-3.8-x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu
LDFLAGS-L+=-lcrypt -lpthread -ldl -lutil -lm -lpython3.8

CFLAGS-W+=-I$(CPYTHONPATH)
CFLAGS-W+=-Wno-unused-result -Wsign-compare -Wformat -Werror=format-security -fwrapv -O2 -Wall -std=c++2a -fopenmp -fconcepts
LDFLAGS-W+=-L$(PYTHONPATH)
LDFLAGS-W+=-lpython39

batch-w: autoencoder-w1 autoencoder-w2 autoencoder-w3 autoencoder-w4 autoencoder-w5 autoencoder-w6 autoencoder-w7 autoencoder-w8 autoencoder-w9 autoencoder-w10 autoencoder-w11 autoencoder-w12

all-w: tuner-w solver-w autoencoder-w estimator-w trainer-w
	
all-l: tuner-l solver-l autoencoder-l estimator-l trainer-l

tuner-w:
	g++ $(CFLAGS-W) src/Finite_Difference_Solver.h src/Finite_Difference_Solver.cpp src/Tune_Params.cpp -o bin/tuner.exe $(LDFLAGS-W)
solver-w:
	g++ $(CFLAGS-W) src/Finite_Difference_Solver.h src/Finite_Difference_Solver.cpp src/Run_Solver.cpp -o bin/solver.exe $(LDFLAGS-W)
autoencoder-w1:
	g++ $(CFLAGS-W) src/Finite_Difference_Solver1.h src/Finite_Difference_Solver1.cpp src/Get_AE_Data1.cpp -o bin/gen_ae_data1.exe $(LDFLAGS-W)
autoencoder-w2:
	g++ $(CFLAGS-W) src/Finite_Difference_Solver2.h src/Finite_Difference_Solver2.cpp src/Get_AE_Data2.cpp -o bin/gen_ae_data2.exe $(LDFLAGS-W)
autoencoder-w3:
	g++ $(CFLAGS-W) src/Finite_Difference_Solver3.h src/Finite_Difference_Solver3.cpp src/Get_AE_Data3.cpp -o bin/gen_ae_data3.exe $(LDFLAGS-W)
autoencoder-w4:
	g++ $(CFLAGS-W) src/Finite_Difference_Solver4.h src/Finite_Difference_Solver4.cpp src/Get_AE_Data4.cpp -o bin/gen_ae_data4.exe $(LDFLAGS-W)
autoencoder-w5:
	g++ $(CFLAGS-W) src/Finite_Difference_Solver5.h src/Finite_Difference_Solver5.cpp src/Get_AE_Data5.cpp -o bin/gen_ae_data5.exe $(LDFLAGS-W)
autoencoder-w6:
	g++ $(CFLAGS-W) src/Finite_Difference_Solver6.h src/Finite_Difference_Solver6.cpp src/Get_AE_Data6.cpp -o bin/gen_ae_data6.exe $(LDFLAGS-W)
autoencoder-w7:
	g++ $(CFLAGS-W) src/Finite_Difference_Solver7.h src/Finite_Difference_Solver7.cpp src/Get_AE_Data7.cpp -o bin/gen_ae_data7.exe $(LDFLAGS-W)
autoencoder-w8:
	g++ $(CFLAGS-W) src/Finite_Difference_Solver8.h src/Finite_Difference_Solver8.cpp src/Get_AE_Data8.cpp -o bin/gen_ae_data8.exe $(LDFLAGS-W)
autoencoder-w9:
	g++ $(CFLAGS-W) src/Finite_Difference_Solver9.h src/Finite_Difference_Solver9.cpp src/Get_AE_Data9.cpp -o bin/gen_ae_data9.exe $(LDFLAGS-W)
autoencoder-w10:
	g++ $(CFLAGS-W) src/Finite_Difference_Solver10.h src/Finite_Difference_Solver10.cpp src/Get_AE_Data10.cpp -o bin/gen_ae_data10.exe $(LDFLAGS-W)
autoencoder-w11:
	g++ $(CFLAGS-W) src/Finite_Difference_Solver11.h src/Finite_Difference_Solver11.cpp src/Get_AE_Data11.cpp -o bin/gen_ae_data11.exe $(LDFLAGS-W)
autoencoder-w12:
	g++ $(CFLAGS-W) src/Finite_Difference_Solver12.h src/Finite_Difference_Solver12.cpp src/Get_AE_Data12.cpp -o bin/gen_ae_data12.exe $(LDFLAGS-W)
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