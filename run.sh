cd src
read -p "simple(s) or real(r) or best(b)? " ch
case $ch in
    [Ss]* ) mode=simple;;
    [Rr]* ) mode=real;;
    [Bb]* ) mode=best;;
    * ) echo "Invalid input"; exit 0;;
esac
python2 train.py $mode
echo "Program terminated."
