CFLAGS-L+=-I/usr/include/python3.8 -I/usr/include/python3.8
CFLAGS-L+=-Wno-unused-result -Wsign-compare -specs=/usr/share/dpkg/no-pie-compile.specs -fstack-protector -Wformat -Werror=format-security -fwrapv -O2 -Wall -std=c++2a -fopenmp -fPIE -fconcepts
LDFLAGS-L+=-L/usr/lib/python3.8/config-3.8-x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu
LDFLAGS-L+=-lcrypt -lpthread -ldl -lutil -lm -lpython3.8

CFLAGS-W+=-I$(CPYTHONPATH)
CFLAGS-W+=-Wno-unused-result -Wsign-compare -Wformat -Werror=format-security -fwrapv -O2 -Wall -std=c++2a -fopenmp -fconcepts
LDFLAGS-W+=-L$(PYTHONPATH)
LDFLAGS-W+=-lpython39

all-l: 
	g++ $(CFLAGS-L) Finite_Difference_Solver.h Finite_Difference_Solver.cpp Run_Solver.cpp -o run_solver.so $(LDFLAGS-L)
	g++ $(CFLAGS-L) Finite_Difference_Solver.h Finite_Difference_Solver.cpp Get_AE_Data.cpp -o get_ae_data.so $(LDFLAGS-L)
	g++ $(CFLAGS-L) Finite_Difference_Solver.h Finite_Difference_Solver.cpp Train_Agent.cpp -o train_agent.so $(LDFLAGS-L)
	
all-w: 
	g++ $(CFLAGS-W) Finite_Difference_Solver.h Finite_Difference_Solver.cpp Run_Solver.cpp -o run_solver.exe $(LDFLAGS-W)
	g++ $(CFLAGS-W) Finite_Difference_Solver.h Finite_Difference_Solver.cpp Get_AE_Data.cpp -o get_ae_data.exe $(LDFLAGS-W)
	g++ $(CFLAGS-W) Finite_Difference_Solver.h Finite_Difference_Solver.cpp Train_Agent.cpp -o train_agent.exe $(LDFLAGS-W)
	
clean-l:
	rm -rf run_solver.so
	rm -rf get_ae_data.so
	rm -rf train_agent.so
	
clean-w:
	del /f run_solver.exe
	del /f get_ae_data.exe
	del /f train_agent.exe