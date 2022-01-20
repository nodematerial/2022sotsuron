str=`basename ${PWD%/*}`

for file in *
do
    case $file in
        yes1_*)
            rm "$file"
            ;;
        *.jpg)
            mv "$file" yes1_"$file"
            ;;
    esac
done