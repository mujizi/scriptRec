#!/bin/bash

# API服务重启脚本
# 用于重启所有推荐API服务
# 支持跨平台使用，自动检测项目根目录

# 获取脚本所在目录作为项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$SCRIPT_DIR"
RECOMMENDATION_DIR="$WORK_DIR/src/recommendation"
LOG_DIR="$WORK_DIR/logs"

# 创建日志目录
mkdir -p "$LOG_DIR"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_message() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

print_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

print_info() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# 检查并杀死指定端口的进程
kill_process_on_port() {
    local port=$1
    local service_name=$2
    
    print_info "检查端口 $port 上的 $service_name 服务..."
    
    # 查找占用端口的进程
    local pids=$(lsof -ti:$port 2>/dev/null)
    
    if [ -n "$pids" ]; then
        print_warning "发现端口 $port 被占用，进程ID: $pids"
        print_info "正在杀死进程..."
        
        for pid in $pids; do
            if kill -9 $pid 2>/dev/null; then
                print_message "成功杀死进程 $pid"
            else
                print_error "无法杀死进程 $pid"
            fi
        done
        
        # 等待进程完全退出
        sleep 2
        
        # 再次检查端口是否释放
        if lsof -ti:$port >/dev/null 2>&1; then
            print_error "端口 $port 仍然被占用"
            return 1
        else
            print_message "端口 $port 已释放"
        fi
    else
        print_info "端口 $port 未被占用"
    fi
    
    return 0
}

# 启动API服务
start_api_service() {
    local script_name=$1
    local port=$2
    local service_name=$3
    local log_file="$LOG_DIR/${script_name%.*}_$(date '+%Y%m%d_%H%M%S').log"
    
    print_info "启动 $service_name 服务 (端口: $port)..."
    
    # 切换到推荐目录
    cd "$RECOMMENDATION_DIR"
    
    # 启动服务
    nohup python "$script_name" > "$log_file" 2>&1 &
    local pid=$!
    
    # 等待服务启动
    sleep 3
    
    # 检查服务是否成功启动
    if lsof -ti:$port >/dev/null 2>&1; then
        print_message "$service_name 服务启动成功 (PID: $pid, 端口: $port)"
        print_info "日志文件: $log_file"
    else
        print_error "$service_name 服务启动失败"
        return 1
    fi
    
    return 0
}

# 主函数
main() {
    print_message "开始重启API服务..."
    print_info "项目根目录: $WORK_DIR"
    
    # 检查工作目录
    if [ ! -d "$WORK_DIR" ]; then
        print_error "工作目录不存在: $WORK_DIR"
        exit 1
    fi
    
    # 检查推荐目录
    if [ ! -d "$RECOMMENDATION_DIR" ]; then
        print_error "推荐目录不存在: $RECOMMENDATION_DIR"
        print_error "请确保脚本位于项目根目录下"
        exit 1
    fi
    
    # 定义服务配置
    declare -A services=(
        ["mcp_chara_api.py"]="7012:人物推荐API"
        ["mcp_scene_api.py"]="7013:场景推荐API"
        ["mcp_script_api.py"]="7014:剧本推荐API"
        ["recommendation_api.py"]="7003:综合推荐API"
    )
    
    # 停止所有服务
    print_info "正在停止所有服务..."
    for script in "${!services[@]}"; do
        IFS=':' read -r port service_name <<< "${services[$script]}"
        kill_process_on_port "$port" "$service_name"
    done
    
    # 等待所有进程完全退出
    sleep 3
    
    # 启动所有服务
    print_info "正在启动所有服务..."
    for script in "${!services[@]}"; do
        IFS=':' read -r port service_name <<< "${services[$script]}"
        if ! start_api_service "$script" "$port" "$service_name"; then
            print_error "启动 $service_name 失败"
        fi
    done
    
    # 最终状态检查
    print_info "检查所有服务状态..."
    for script in "${!services[@]}"; do
        IFS=':' read -r port service_name <<< "${services[$script]}"
        if lsof -ti:$port >/dev/null 2>&1; then
            local pid=$(lsof -ti:$port)
            print_message "✓ $service_name 运行中 (PID: $pid, 端口: $port)"
        else
            print_error "✗ $service_name 未运行 (端口: $port)"
        fi
    done
    
    print_message "API服务重启完成！"
    
    # 显示日志文件位置
    print_info "日志文件位置: $LOG_DIR"
    ls -la "$LOG_DIR"/*.log 2>/dev/null | tail -10
}

# 显示帮助信息
show_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help     显示此帮助信息"
    echo "  -s, --status   只检查服务状态"
    echo "  -k, --kill     只停止所有服务"
    echo ""
    echo "默认行为: 停止所有服务并重新启动"
    echo ""
    echo "说明:"
    echo "  - 脚本会自动检测项目根目录"
    echo "  - 所有日志文件将保存在 logs/ 目录下"
    echo "  - 支持的服务端口: 7003(综合推荐), 7012(人物), 7013(场景), 7014(剧本)"
}

# 检查服务状态
check_status() {
    print_info "检查所有API服务状态..."
    
    declare -A services=(
        ["mcp_chara_api.py"]="7012:人物推荐API"
        ["mcp_scene_api.py"]="7013:场景推荐API"
        ["mcp_script_api.py"]="7014:剧本推荐API"
        ["recommendation_api.py"]="7003:综合推荐API"
    )
    
    for script in "${!services[@]}"; do
        IFS=':' read -r port service_name <<< "${services[$script]}"
        if lsof -ti:$port >/dev/null 2>&1; then
            local pid=$(lsof -ti:$port)
            print_message "✓ $service_name 运行中 (PID: $pid, 端口: $port)"
        else
            print_error "✗ $service_name 未运行 (端口: $port)"
        fi
    done
}

# 只停止服务
kill_all_services() {
    print_info "停止所有API服务..."
    
    declare -A services=(
        ["mcp_chara_api.py"]="7012:人物推荐API"
        ["mcp_scene_api.py"]="7013:场景推荐API"
        ["mcp_script_api.py"]="7014:剧本推荐API"
        ["recommendation_api.py"]="7003:综合推荐API"
    )
    
    for script in "${!services[@]}"; do
        IFS=':' read -r port service_name <<< "${services[$script]}"
        kill_process_on_port "$port" "$service_name"
    done
    
    print_message "所有服务已停止"
}

# 解析命令行参数
case "${1:-}" in
    -h|--help)
        show_help
        exit 0
        ;;
    -s|--status)
        check_status
        exit 0
        ;;
    -k|--kill)
        kill_all_services
        exit 0
        ;;
    "")
        main
        ;;
    *)
        print_error "未知选项: $1"
        show_help
        exit 1
        ;;
esac 