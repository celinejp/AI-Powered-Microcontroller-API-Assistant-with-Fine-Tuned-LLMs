#!/bin/bash

# Microcontroller API Assistant - Complete Project Runner
# This script starts both backend and frontend services with proper setup

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_header() {
    echo -e "${PURPLE}🚀 $1${NC}"
}

print_success() {
    echo -e "${CYAN}🎉 $1${NC}"
}

# Function to check if a port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function to kill process on port
kill_port() {
    local port=$1
    local pid=$(lsof -ti:$port)
    if [ ! -z "$pid" ]; then
        echo "Killing process $pid on port $port"
        kill -9 $pid 2>/dev/null || true
        sleep 1
    fi
}

# Function to wait for service to be ready
wait_for_service() {
    local url=$1
    local service_name=$2
    local max_attempts=30
    local attempt=1
    
    print_info "Waiting for $service_name to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$url" >/dev/null 2>&1; then
            print_status "$service_name is ready!"
            return 0
        fi
        
        echo -n "."
        sleep 1
        attempt=$((attempt + 1))
    done
    
    print_error "$service_name failed to start within $max_attempts seconds"
    return 1
}

# Function to setup environment
setup_environment() {
    print_header "Setting up environment..."
    
    # Check if .env files exist, create if not
    if [ ! -f "backend/.env" ]; then
        print_info "Creating backend/.env from template..."
        cp backend/env.example backend/.env
        print_status "Backend .env created"
    fi
    
    if [ ! -f "frontend/.env" ]; then
        print_info "Creating frontend/.env from template..."
        cp frontend/env.example frontend/.env
        print_status "Frontend .env created"
    fi
    
    # Check if virtual environment exists
    if [ ! -d "backend/venv" ]; then
        print_warning "Backend virtual environment not found. Creating one..."
        cd backend
        python3 -m venv venv
        source venv/bin/activate
        pip install --upgrade pip
        pip install -r requirements.txt 2>/dev/null || pip install fastapi uvicorn transformers torch
        cd ..
        print_status "Backend virtual environment created"
    fi
    
    # Check if node_modules exists
    if [ ! -d "frontend/node_modules" ]; then
        print_warning "Frontend dependencies not found. Installing..."
        cd frontend
        npm install
        cd ..
        print_status "Frontend dependencies installed"
    fi
}

# Function to start backend
start_backend() {
    print_header "Starting Backend Service..."
    
    # Kill any existing process on port 8000
    if check_port 8000; then
        print_warning "Port 8000 is in use. Stopping existing process..."
        kill_port 8000
    fi
    
    # Start backend
    cd backend
    source venv/bin/activate
    
    # Set environment variables for better compatibility
    export PYTHONPATH="${PYTHONPATH}:$(pwd)"
    export HF_HOME="${HOME}/.cache/huggingface"
    
    print_info "Starting FastAPI server on port 8000..."
    nohup python main.py > ../logs/backend.log 2>&1 &
    BACKEND_PID=$!
    echo $BACKEND_PID > ../logs/backend.pid
    
    cd ..
    
    # Wait for backend to be ready
    if wait_for_service "http://localhost:8000/health" "Backend"; then
        print_status "Backend started successfully (PID: $BACKEND_PID)"
        return 0
    else
        print_error "Backend failed to start"
        return 1
    fi
}

# Function to start frontend
start_frontend() {
    print_header "Starting Frontend Service..."
    
    # Kill any existing process on port 3000
    if check_port 3000; then
        print_warning "Port 3000 is in use. Stopping existing process..."
        kill_port 3000
    fi
    
    # Start frontend
    cd frontend
    
    print_info "Starting React development server on port 3000..."
    nohup npm start > ../logs/frontend.log 2>&1 &
    FRONTEND_PID=$!
    echo $FRONTEND_PID > ../logs/frontend.pid
    
    cd ..
    
    # Wait for frontend to be ready
    if wait_for_service "http://localhost:3000" "Frontend"; then
        print_status "Frontend started successfully (PID: $FRONTEND_PID)"
        return 0
    else
        print_error "Frontend failed to start"
        return 1
    fi
}

# Function to show status
show_status() {
    print_header "Service Status"
    echo "=================="
    
    # Check backend
    if check_port 8000; then
        print_status "Backend: Running on http://localhost:8000"
        curl -s http://localhost:8000/health | jq . 2>/dev/null || echo "Backend health check failed"
    else
        print_error "Backend: Not running"
    fi
    
    echo ""
    
    # Check frontend
    if check_port 3000; then
        print_status "Frontend: Running on http://localhost:3000"
    else
        print_error "Frontend: Not running"
    fi
    
    echo ""
    print_info "API Documentation: http://localhost:8000/docs"
    print_info "Web Application: http://localhost:3000"
}

# Function to stop all services
stop_services() {
    print_header "Stopping all services..."
    
    # Stop backend
    if [ -f "logs/backend.pid" ]; then
        BACKEND_PID=$(cat logs/backend.pid)
        if kill -0 $BACKEND_PID 2>/dev/null; then
            kill $BACKEND_PID
            print_status "Backend stopped (PID: $BACKEND_PID)"
        fi
        rm -f logs/backend.pid
    fi
    
    # Stop frontend
    if [ -f "logs/frontend.pid" ]; then
        FRONTEND_PID=$(cat logs/frontend.pid)
        if kill -0 $FRONTEND_PID 2>/dev/null; then
            kill $FRONTEND_PID
            print_status "Frontend stopped (PID: $FRONTEND_PID)"
        fi
        rm -f logs/frontend.pid
    fi
    
    # Kill any remaining processes on ports
    kill_port 8000
    kill_port 3000
    
    print_status "All services stopped"
}

# Function to show logs
show_logs() {
    local service=$1
    
    if [ -z "$service" ]; then
        print_info "Available logs:"
        echo "  backend  - Backend service logs"
        echo "  frontend - Frontend service logs"
        echo "  all      - All logs"
        echo ""
        echo "Usage: $0 logs [service]"
        return
    fi
    
    case $service in
        "backend")
            if [ -f "logs/backend.log" ]; then
                print_info "Backend logs:"
                tail -f logs/backend.log
            else
                print_error "Backend log file not found"
            fi
            ;;
        "frontend")
            if [ -f "logs/frontend.log" ]; then
                print_info "Frontend logs:"
                tail -f logs/frontend.log
            else
                print_error "Frontend log file not found"
            fi
            ;;
        "all")
            print_info "All logs:"
            if [ -f "logs/backend.log" ]; then
                echo "=== BACKEND LOGS ==="
                tail -20 logs/backend.log
                echo ""
            fi
            if [ -f "logs/frontend.log" ]; then
                echo "=== FRONTEND LOGS ==="
                tail -20 logs/frontend.log
            fi
            ;;
        *)
            print_error "Unknown service: $service"
            ;;
    esac
}

# Function to run tests
run_tests() {
    print_header "Running tests..."
    
    if [ -f "test_api.py" ]; then
        print_info "Running API tests..."
        python test_api.py
    else
        print_warning "test_api.py not found"
    fi
    
    if [ -f "test_curl.sh" ]; then
        print_info "Running curl tests..."
        ./test_curl.sh
    else
        print_warning "test_curl.sh not found"
    fi
}

# Main function
main() {
    # Create logs directory
    mkdir -p logs
    
    # Parse command line arguments
    case "${1:-start}" in
        "start")
            print_header "Microcontroller API Assistant - Starting All Services"
            echo "=========================================================="
            
            setup_environment
            
            if start_backend && start_frontend; then
                echo ""
                print_success "All services started successfully!"
                echo ""
                show_status
                echo ""
                print_info "Press Ctrl+C to stop all services"
                
                # Keep script running and handle Ctrl+C
                trap 'echo ""; stop_services; exit 0' INT
                while true; do
                    sleep 1
                done
            else
                print_error "Failed to start services"
                exit 1
            fi
            ;;
        "stop")
            stop_services
            ;;
        "restart")
            stop_services
            sleep 2
            $0 start
            ;;
        "status")
            show_status
            ;;
        "logs")
            show_logs "$2"
            ;;
        "test")
            run_tests
            ;;
        "setup")
            setup_environment
            print_success "Environment setup completed!"
            ;;
        "help"|"-h"|"--help")
            echo "Microcontroller API Assistant - Project Runner"
            echo "=============================================="
            echo ""
            echo "Usage: $0 [command]"
            echo ""
            echo "Commands:"
            echo "  start    - Start all services (default)"
            echo "  stop     - Stop all services"
            echo "  restart  - Restart all services"
            echo "  status   - Show service status"
            echo "  logs     - Show logs (backend|frontend|all)"
            echo "  test     - Run test suite"
            echo "  setup    - Setup environment only"
            echo "  help     - Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                # Start all services"
            echo "  $0 start          # Start all services"
            echo "  $0 stop           # Stop all services"
            echo "  $0 status         # Check service status"
            echo "  $0 logs backend   # View backend logs"
            echo "  $0 test           # Run tests"
            ;;
        *)
            print_error "Unknown command: $1"
            echo "Use '$0 help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
