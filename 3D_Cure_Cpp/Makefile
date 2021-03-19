CXX = g++
CXXFLAGS = -fopenmp -Wall
INC = -IC:\Users\GKSch\AppData\Local\Programs\Python\Python39\include
LIB_PATH= -LC:\Users\GKSch\AppData\Local\Programs\Python\Python39
LIB = -lpython39
NAME = FES

auto: 
	$(CXX) $(INC) $(LIB_PATH) $(LIB) $(CXXFLAGS) -O2 Finite_Element_Solver.h Finite_Element_Solver.cpp Train_Autoencoder.cpp -o $(NAME)
	
ppo: 
	$(CXX) $(INC) $(LIB_PATH) $(LIB) $(CXXFLAGS) -O2 Finite_Element_Solver.h Finite_Element_Solver.cpp Train_Controller.cpp -o $(NAME)
	
ppo_debug: 
	$(CXX) $(INC) $(LIB_PATH) $(LIB) $(CXXFLAGS) -g Finite_Element_Solver.h Finite_Element_Solver.cpp Train_Controller.cpp -o $(NAME)