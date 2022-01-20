str=`basename ${PWD%/*}`

for file in *
do
    case $file in
        yes2_*)
            rm "$file"
            ;;
    esac
done

for file in *
do
    case $file in
        *.jpg)
            mv "$file" yes2_"$file"
            ;;
    esac
done