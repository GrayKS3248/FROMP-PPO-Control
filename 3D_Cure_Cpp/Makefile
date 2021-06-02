CXX = g++
CXXFLAGS = -fopenmp -Wall -ansi -std=c++2a
INC = -I$(CPYTHONPATH)
LIB_PATH= -L$(PYTHONPATH)
LIB = -lpython39

data: 
	$(CXX) $(INC) $(LIB_PATH) $(LIB) -O2 $(CXXFLAGS) Finite_Element_Solver.h Finite_Element_Solver.cpp Get_Training_Data.cpp -o get_data

trainrk2t: 
	$(CXX) $(INC) $(LIB_PATH) $(LIB) -O2 $(CXXFLAGS) Finite_Element_Solver.h Finite_Element_Solver_RK2T_Hybrid.cpp Train_Agent.cpp -o train_agent

trainrk4: 
	$(CXX) $(INC) $(LIB_PATH) $(LIB) -O2 $(CXXFLAGS) Finite_Element_Solver.h Finite_Element_Solver_RK4.cpp Train_Agent.cpp -o train_agent
	
train: 
	$(CXX) $(INC) $(LIB_PATH) $(LIB) -O2 $(CXXFLAGS) Finite_Element_Solver.h Finite_Element_Solver.cpp Train_Agent.cpp -o train_agent