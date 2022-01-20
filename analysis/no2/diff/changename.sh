str=`basename ${PWD%/*}`

for file in *
do
    case $file in
        no2_*)
            rm "$file"
            ;;
    esac
done

for file in *
do
    case $file in
        *.jpg)
            mv "$file" no2_"$file"
            ;;
    esac
done