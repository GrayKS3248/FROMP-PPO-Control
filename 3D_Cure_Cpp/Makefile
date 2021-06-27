CFLAGS-L+=-I/usr/include/python3.8 -I/usr/include/python3.8
CFLAGS-L+=-Wno-unused-result -Wsign-compare -specs=/usr/share/dpkg/no-pie-compile.specs -fstack-protector -Wformat -Werror=format-security -fwrapv -O2 -Wall -std=c++2a -fopenmp -fPIE
LDFLAGS-L+=-L/usr/lib/python3.8/config-3.8-x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu
LDFLAGS-L+=-lcrypt -lpthread -ldl -lutil -lm -lpython3.8

CFLAGS-W+=-I$(CPYTHONPATH)
CFLAGS-W+=-Wno-unused-result -Wsign-compare -Wformat -Werror=format-security -fwrapv -O2 -Wall -std=c++2a -fopenmp
LDFLAGS-W+=-L$(PYTHONPATH)
LDFLAGS-W+=-lpython39

data-l: 
	g++ $(CFLAGS-L) Finite_Difference_Solver.h Finite_Difference_Solver.cpp Get_Training_Data.cpp -o get_ae_data.so $(LDFLAGS-L)
	
data-w: 
	g++ $(CFLAGS-W) Finite_Difference_Solver.h Finite_Difference_Solver.cpp Get_Training_Data.cpp -o get_ae_data.exe $(LDFLAGS-W)
	
train-l: 
	g++ $(CFLAGS-L) Finite_Difference_Solver.h Finite_Difference_Solver.cpp Train_Agent.cpp -o train_agent.so $(LDFLAGS-L)
	
train-w: 
	g++ $(CFLAGS-W) Finite_Difference_Solver.h Finite_Difference_Solver.cpp Train_Agent.cpp -o train_agent.exe $(LDFLAGS-W)
	
clean-l:
	rm -rf train_agent.so
	rm -rf get_ae_data.so
	rm -rf train_agent.exe
	rm -rf get_ae_data.exe