# Ignore certain packages conda does
pip list --format=freeze | grep -Po '^((?!mkl).)*$' >requirements.txt
