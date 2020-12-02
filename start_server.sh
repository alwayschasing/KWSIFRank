#!/bin/sh

work_dir=`pwd`
alias techo='echo `date +"%F %T |"`'
restart_wait=2
server_name="multiprocess_extract"
bin_file="${server_name}.py"
conf_file="server_config.json"

cd $work_dir

usage()
{
    echo "$0 start|stop|restart|monitor|status"
}

start()
{
    techo "Start $server_name..."
    nohup python $bin_file  > log/tagclassify.log 2>&1 &
    if [ $? -eq 0 ];then
        techo "Start $server_name OK!"
    else
        techo "Start $server_name FAIL!"
    fi
}

stop()
{
    techo "Stop $server_name..."
    kill -9 `ps aux | grep "$bin_file " | grep -v grep |awk '{print $2}'` > /dev/null 2>&1
    if [ $? -eq 0 ];then
        techo "Stop $server_name OK!"
    else
        techo "Stop $server_name FAIL!"
    fi
}

restart()
{
    stop
    sleep $restart_wait
    start
}

status()
{
    ps aux | grep "$bin_file " | grep -v grep > /dev/null 2>&1
    if [ $? -eq 0 ];then
        techo "$server_name is running"
    else
        techo "$server_name is stoped"
        return 1
    fi
}

monitor()
{
    status
    if [ $? -ne 0 ];then
        start
    fi
}

start
#case "$1" in
#"start")
#    start
#    ;;
#"stop")
#    stop
#    ;;
#"restart")
#    restart $2
#    ;;
#"monitor")
#    monitor
#    ;;
#"status")
#    status
#    ;;
#*)
#    usage
#    ;;
#esac
#exit $?
