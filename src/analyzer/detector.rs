use crate::config::AnalysisConfig;
#[cfg(feature = "yolo")]
use crate::error::AppError;
use crate::error::AppResult;

#[cfg(feature = "yolo")]
use tracing::info;

pub(crate) fn detect_person_confidence(
    detector: &mut Option<YoloDetector>,
    center_bgr: &[u8],
    width: usize,
    height: usize,
    config: &AnalysisConfig,
) -> AppResult<Option<f32>> {
    if !config.enable_yolo {
        return Ok(None);
    }
    if let Some(detector) = detector.as_mut() {
        detector.detect_person_confidence(center_bgr, width, height)
    } else {
        Ok(None)
    }
}

#[cfg(feature = "yolo")]
pub(crate) struct YoloDetector {
    session: ort::session::Session,
    input_w: usize,
    input_h: usize,
    layout: YoloInputLayout,
    scratch: Vec<f32>,
    letterbox_plan: Option<LetterboxPlan>,
}

#[cfg(feature = "yolo")]
#[derive(Clone, Copy)]
enum YoloInputLayout {
    Nchw,
    Nhwc,
}

#[cfg(feature = "yolo")]
struct LetterboxPlan {
    src_w: usize,
    src_h: usize,
    pad_x: usize,
    pad_y: usize,
    source_x: Vec<usize>,
    source_y: Vec<usize>,
}

#[cfg(feature = "yolo")]
impl LetterboxPlan {
    fn new(src_w: usize, src_h: usize, dst_w: usize, dst_h: usize) -> Self {
        let scale = (dst_w as f32 / src_w.max(1) as f32).min(dst_h as f32 / src_h.max(1) as f32);
        let resized_w = ((src_w as f32) * scale).round().max(1.0) as usize;
        let resized_h = ((src_h as f32) * scale).round().max(1.0) as usize;
        let pad_x = dst_w.saturating_sub(resized_w) / 2;
        let pad_y = dst_h.saturating_sub(resized_h) / 2;
        let source_x = (0..resized_w)
            .map(|x| ((x as f32 / scale).floor() as usize).min(src_w.saturating_sub(1)))
            .collect();
        let source_y = (0..resized_h)
            .map(|y| ((y as f32 / scale).floor() as usize).min(src_h.saturating_sub(1)))
            .collect();
        Self {
            src_w,
            src_h,
            pad_x,
            pad_y,
            source_x,
            source_y,
        }
    }

    fn matches(&self, src_w: usize, src_h: usize) -> bool {
        self.src_w == src_w && self.src_h == src_h
    }
}

#[cfg(feature = "yolo")]
impl YoloDetector {
    pub(crate) fn from_config(config: &AnalysisConfig) -> AppResult<Option<Self>> {
        let Some(model_path) = config.yolo_model.as_ref() else {
            return Ok(None);
        };

        use ort::ep;

        let mut eps: Vec<ort::ep::ExecutionProviderDispatch> = Vec::new();
        let mut ep_names: Vec<&str> = Vec::new();

        #[cfg(feature = "tensorrt")]
        {
            eps.push(ep::TensorRT::default().build());
            ep_names.push("TensorRT");
        }
        #[cfg(feature = "cuda")]
        {
            eps.push(ep::CUDA::default().build());
            ep_names.push("CUDA");
        }
        #[cfg(feature = "directml")]
        {
            eps.push(ep::DirectML::default().build());
            ep_names.push("DirectML");
        }
        #[cfg(feature = "coreml")]
        {
            eps.push(ep::CoreML::default().build());
            ep_names.push("CoreML");
        }

        eps.push(ep::CPU::default().build());
        ep_names.push("CPU");

        info!("YOLO EP priority chain: [{}]", ep_names.join(" → "));

        let session = ort::session::Session::builder()
            .map_err(|e| AppError::Inference(e.to_string()))?
            .with_execution_providers(eps)
            .map_err(|e| AppError::Inference(e.to_string()))?
            .with_intra_threads(config.yolo_intra_threads)
            .map_err(|e| AppError::Inference(e.to_string()))?
            .commit_from_file(model_path)
            .map_err(|e| AppError::Inference(e.to_string()))?;

        info!(
            "YOLO session loaded from {}  (intra_threads={})",
            model_path.display(),
            config.yolo_intra_threads,
        );

        let input = session
            .inputs()
            .first()
            .ok_or_else(|| AppError::Unsupported("YOLO model has no inputs".to_string()))?;
        let (layout, input_h, input_w) = infer_yolo_input_shape(input.dtype())?;
        let scratch = vec![114.0f32 / 255.0; 3 * input_h * input_w];
        Ok(Some(Self {
            session,
            input_w,
            input_h,
            layout,
            scratch,
            letterbox_plan: None,
        }))
    }

    pub(crate) fn detect_person_confidence(
        &mut self,
        center_bgr: &[u8],
        width: usize,
        height: usize,
    ) -> AppResult<Option<f32>> {
        use ndarray::ArrayView4;
        use ort::value::TensorRef;

        let expected_len = width
            .checked_mul(height)
            .and_then(|pixels| pixels.checked_mul(3))
            .ok_or_else(|| AppError::Unsupported("YOLO input dimensions overflow".to_string()))?;
        if center_bgr.len() < expected_len {
            return Err(AppError::Unsupported(format!(
                "YOLO input frame is truncated: expected {expected_len} bytes, got {}",
                center_bgr.len()
            )));
        }
        if self
            .letterbox_plan
            .as_ref()
            .is_none_or(|plan| !plan.matches(width, height))
        {
            self.letterbox_plan = Some(LetterboxPlan::new(
                width,
                height,
                self.input_w,
                self.input_h,
            ));
        }
        letterbox_bgr_to_normalized(
            &mut self.scratch,
            center_bgr,
            self.input_w,
            self.input_h,
            self.layout,
            self.letterbox_plan
                .as_ref()
                .expect("plan initialized above"),
        );

        let outputs = match self.layout {
            YoloInputLayout::Nchw => {
                let view =
                    ArrayView4::from_shape((1, 3, self.input_h, self.input_w), &self.scratch)
                        .map_err(|e| AppError::Inference(e.to_string()))?;
                let input = TensorRef::from_array_view(view)
                    .map_err(|e| AppError::Inference(e.to_string()))?;
                self.session
                    .run(ort::inputs![input])
                    .map_err(|e| AppError::Inference(e.to_string()))?
            }
            YoloInputLayout::Nhwc => {
                let view =
                    ArrayView4::from_shape((1, self.input_h, self.input_w, 3), &self.scratch)
                        .map_err(|e| AppError::Inference(e.to_string()))?;
                let input = TensorRef::from_array_view(view)
                    .map_err(|e| AppError::Inference(e.to_string()))?;
                self.session
                    .run(ort::inputs![input])
                    .map_err(|e| AppError::Inference(e.to_string()))?
            }
        };

        if outputs.len() == 0 {
            return Err(AppError::Unsupported(
                "YOLO produced no outputs".to_string(),
            ));
        }
        let output = &outputs[0];
        let view = output
            .try_extract_array::<f32>()
            .map_err(|e| AppError::Inference(e.to_string()))?;
        Ok(best_person_confidence(view.into_dyn()))
    }
}

#[cfg(not(feature = "yolo"))]
pub(crate) struct YoloDetector;

#[cfg(not(feature = "yolo"))]
impl YoloDetector {
    pub(crate) fn from_config(_config: &AnalysisConfig) -> AppResult<Option<Self>> {
        Ok(None)
    }

    fn detect_person_confidence(
        &mut self,
        _center_bgr: &[u8],
        _width: usize,
        _height: usize,
    ) -> AppResult<Option<f32>> {
        Ok(None)
    }
}

#[cfg(feature = "yolo")]
fn infer_yolo_input_shape(
    dtype: &ort::value::ValueType,
) -> AppResult<(YoloInputLayout, usize, usize)> {
    let dims = dtype
        .tensor_shape()
        .ok_or_else(|| AppError::Unsupported("YOLO input is not a tensor".to_string()))?;
    if dims.len() != 4 {
        return Err(AppError::Unsupported(format!(
            "YOLO input must be 4D, got {} dimensions",
            dims.len()
        )));
    }

    let d = dims.iter().copied().collect::<Vec<_>>();
    let positive = |v: i64| if v > 0 { Some(v as usize) } else { None };

    if d[1] == 3 {
        return Ok((
            YoloInputLayout::Nchw,
            positive(d[2]).unwrap_or(640),
            positive(d[3]).unwrap_or(640),
        ));
    }
    if d[3] == 3 {
        return Ok((
            YoloInputLayout::Nhwc,
            positive(d[1]).unwrap_or(640),
            positive(d[2]).unwrap_or(640),
        ));
    }

    Err(AppError::Unsupported(format!(
        "unsupported YOLO input layout: {:?}",
        d
    )))
}

#[cfg(feature = "yolo")]
fn letterbox_bgr_to_normalized(
    out: &mut [f32],
    source_bgr: &[u8],
    dst_w: usize,
    dst_h: usize,
    layout: YoloInputLayout,
    plan: &LetterboxPlan,
) {
    let fill = 114.0f32 / 255.0;
    let normalize = 1.0f32 / 255.0;

    out.fill(fill);

    match layout {
        YoloInputLayout::Nchw => {
            let plane = dst_h * dst_w;
            for (y, &src_y) in plan.source_y.iter().enumerate() {
                let src_y_off = src_y * plan.src_w;
                let dst_y_off = (y + plan.pad_y) * dst_w + plan.pad_x;

                for (x, &src_x) in plan.source_x.iter().enumerate() {
                    let src_idx = (src_y_off + src_x) * 3;
                    let dst_idx = dst_y_off + x;

                    out[dst_idx] = source_bgr[src_idx + 2] as f32 * normalize; // R
                    out[plane + dst_idx] = source_bgr[src_idx + 1] as f32 * normalize; // G
                    out[2 * plane + dst_idx] = source_bgr[src_idx] as f32 * normalize; // B
                }
            }
        }
        YoloInputLayout::Nhwc => {
            for (y, &src_y) in plan.source_y.iter().enumerate() {
                let src_y_off = src_y * plan.src_w;
                let dst_y_off = ((y + plan.pad_y) * dst_w + plan.pad_x) * 3;

                for (x, &src_x) in plan.source_x.iter().enumerate() {
                    let src_idx = (src_y_off + src_x) * 3;
                    let dst_idx = dst_y_off + x * 3;

                    out[dst_idx] = source_bgr[src_idx + 2] as f32 * normalize;
                    out[dst_idx + 1] = source_bgr[src_idx + 1] as f32 * normalize;
                    out[dst_idx + 2] = source_bgr[src_idx] as f32 * normalize;
                }
            }
        }
    }
}

#[cfg(feature = "yolo")]
fn best_person_confidence(output: ndarray::ArrayViewD<'_, f32>) -> Option<f32> {
    use ndarray::Axis;

    match output.ndim() {
        3 => {
            if output.shape()[0] == 0 {
                return None;
            }
            best_person_confidence_2d(
                output
                    .index_axis(Axis(0), 0)
                    .into_dimensionality::<ndarray::Ix2>()
                    .ok()?,
            )
        }
        2 => best_person_confidence_2d(output.into_dimensionality::<ndarray::Ix2>().ok()?),
        _ => None,
    }
}

#[cfg(feature = "yolo")]
pub(crate) fn best_person_confidence_2d(output: ndarray::ArrayView2<'_, f32>) -> Option<f32> {
    let rows = output.shape()[0];
    let cols = output.shape()[1];

    let is_attrs_rows = if rows >= 5 && cols >= 5 {
        rows < cols
    } else {
        rows >= 5
    };

    if is_attrs_rows {
        let mut best = 0.0f32;
        // YOLOv5/v7 style: [85, N], or pose-style [6, N].
        if rows == 6 && looks_like_post_nms(output.row(5).iter().copied()) {
            let score = output.row(4);
            let class = output.row(5);
            for i in 0..cols {
                if class[i].round() as i32 == 0 {
                    best = best.max(score[i].max(0.0));
                }
            }
        } else if rows >= 85 || rows == 6 {
            let obj = output.row(4);
            let p1 = output.row(5);
            for i in 0..cols {
                best = best.max((obj[i] * p1[i]).max(0.0));
            }
        } else if rows >= 5 {
            // YOLOv8 style: [5, 8400] (no objectness, person is index 4)
            let p1 = output.row(4);
            for i in 0..cols {
                best = best.max(p1[i].max(0.0));
            }
        }
        Some(best)
    } else {
        let mut best = 0.0f32;
        // YOLOv5/v7 style: [N, 85], or pose-style [N, 6].
        if cols == 6 && looks_like_post_nms(output.column(5).iter().copied()) {
            let score = output.column(4);
            let class = output.column(5);
            for i in 0..rows {
                if class[i].round() as i32 == 0 {
                    best = best.max(score[i].max(0.0));
                }
            }
        } else if cols >= 85 || cols == 6 {
            let obj = output.column(4);
            let p1 = output.column(5);
            for i in 0..rows {
                best = best.max((obj[i] * p1[i]).max(0.0));
            }
        } else if cols >= 5 {
            // YOLOv8 style: [8400, 5]
            let p1 = output.column(4);
            for i in 0..rows {
                best = best.max(p1[i].max(0.0));
            }
        } else {
            return None;
        }
        Some(best)
    }
}

#[cfg(feature = "yolo")]
fn looks_like_post_nms(classes: impl Iterator<Item = f32>) -> bool {
    let mut seen = false;
    for class in classes {
        if !class.is_finite() || class < 0.0 || (class - class.round()).abs() > 1e-4 {
            return false;
        }
        seen = true;
    }
    seen
}
