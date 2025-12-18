@echo off
echo ================================================
echo 电信诈骗犯罪网络关键节点识别研究 - 完整运行脚本
echo ================================================
echo.

cd /d "%~dp0"

echo [1/7] 正在生成电信诈骗网络数据集...
echo ---------------------------------------------
python data_generation.py
if errorlevel 1 (
    echo 数据生成失败！
    pause
    exit /b 1
)
echo 数据生成完成！
echo.

echo [2/7] 正在进行网络基础分析...
echo ---------------------------------------------
python network_analysis.py
if errorlevel 1 (
    echo 基础分析失败！
    pause
    exit /b 1
)
echo 基础分析完成！
echo.

echo [3/7] 正在进行复杂网络特性分析...
echo ---------------------------------------------
python complex_network_analysis.py
if errorlevel 1 (
    echo 复杂网络分析失败！
    pause
    exit /b 1
)
echo 复杂网络分析完成！
echo.

echo [4/7] 正在进行社会网络特性分析...
echo ---------------------------------------------
python social_network_analysis.py
if errorlevel 1 (
    echo 社会网络分析失败！
    pause
    exit /b 1
)
echo 社会网络分析完成！
echo.

echo [5/7] 正在运行关键节点识别算法...
echo ---------------------------------------------
python key_node_algorithm.py
if errorlevel 1 (
    echo 关键节点算法失败！
    pause
    exit /b 1
)
echo 关键节点算法完成！
echo.

echo [6/7] 正在进行算法验证与测试...
echo ---------------------------------------------
python algorithm_validation.py
if errorlevel 1 (
    echo 算法验证失败！
    pause
    exit /b 1
)
echo 算法验证完成！
echo.

echo [7/7] 正在生成研究报告...
echo ---------------------------------------------
python research_report.py
if errorlevel 1 (
    echo 报告生成失败！
    pause
    exit /b 1
)
echo 报告生成完成！
echo.

echo ================================================
echo 所有分析流程执行完毕！
echo.
echo 生成的文件：
echo - data/                        原始数据集
echo - visualizations/              基础可视化图表
echo - complex_analysis/           复杂网络分析结果
echo - social_analysis/            社会网络分析结果
echo - algorithm_analysis/         算法分析结果
echo - validation_results/         算法验证结果
echo - research_report.docx        完整研究报告
echo.
echo ================================================
echo 研究完成！按任意键退出...
pause
