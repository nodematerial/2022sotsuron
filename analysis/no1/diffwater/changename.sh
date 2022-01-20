str=`basename ${PWD%/*}`

for file in *
do
    case $file in
        no1_*)
            rm "$file"
            ;;
    esac
done

for file in *
do
    case $file in
        *.jpg)
            mv "$file" no1_"$file"
            ;;
    esac
done