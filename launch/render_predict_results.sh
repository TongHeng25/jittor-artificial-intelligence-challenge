predict_output_dir="predict"
render_output_dir="tmp_predict"
render="true"
export_fbx="false"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --predict_output_dir) predict_output_dir="$2"; shift ;;
        --render_output_dir) render_output_dir="$2"; shift ;;
        --render) render="$2"; shift ;;
        --export_fbx) export_fbx="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

cmd=" \
    python render_predict_results.py \
    --predict_output_dir $predict_output_dir \
    --render_output_dir $render_output_dir \
    --render $render \
    --export_fbx $export_fbx \
"

cmd="$cmd &"
eval $cmd

wait