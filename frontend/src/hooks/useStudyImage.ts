import { useEffect, useState } from "react";
import { apiUrl } from "@/lib/apiConfig";

interface State {
  image: HTMLImageElement | null;
  isLoading: boolean;
  error: Error | null;
}

export function useStudyImage(studyId: string | undefined): State {
  const [state, setState] = useState<State>({
    image: null,
    isLoading: Boolean(studyId),
    error: null,
  });

  useEffect(() => {
    if (!studyId) {
      setState({ image: null, isLoading: false, error: null });
      return;
    }
    setState({ image: null, isLoading: true, error: null });
    const img = new Image();
    img.decoding = "async";
    img.src = apiUrl(`/api/v1/studies/${studyId}/image`);
    let cancelled = false;
    img.onload = () => {
      if (!cancelled) setState({ image: img, isLoading: false, error: null });
    };
    img.onerror = () => {
      if (!cancelled) {
        setState({
          image: null,
          isLoading: false,
          error: new Error("Не удалось загрузить изображение исследования"),
        });
      }
    };
    return () => {
      cancelled = true;
      img.onload = null;
      img.onerror = null;
    };
  }, [studyId]);

  return state;
}
