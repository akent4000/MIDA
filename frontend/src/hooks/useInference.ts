import { useEffect, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
  InferenceService,
  type InferenceResultPublic,
  type InferenceSubmitResponse,
  type ModelListItem,
  ModelsService,
} from "@/api";
import { apiUrl, wsUrl } from "@/lib/apiConfig";

export function useModels() {
  return useQuery<ModelListItem[]>({
    queryKey: ["models"],
    queryFn: () => ModelsService.listModelsApiV1ModelsGet(),
    staleTime: 5 * 60_000,
  });
}

interface RunArgs {
  studyId: string;
  toolId: string;
}

export function useRunInference() {
  return useMutation<InferenceSubmitResponse, unknown, RunArgs>({
    mutationFn: ({ studyId, toolId }) =>
      InferenceService.submitInferenceApiV1StudiesStudyIdInferencePost({
        studyId,
        requestBody: { tool_id: toolId },
      }),
  });
}

export type TaskStatus = "idle" | "queued" | "running" | "done" | "failed";

export interface TaskState {
  status: TaskStatus;
  message: string | null;
}

export function useTaskStatus(taskId: string | undefined): TaskState {
  const [state, setState] = useState<TaskState>({ status: "idle", message: null });

  useEffect(() => {
    if (!taskId) {
      setState({ status: "idle", message: null });
      return;
    }
    setState({ status: "queued", message: null });

    let ws: WebSocket | null = null;
    try {
      ws = new WebSocket(wsUrl(`/ws/tasks/${taskId}`));
    } catch {
      setState({ status: "failed", message: "WebSocket недоступен" });
      return;
    }

    ws.onmessage = (ev) => {
      try {
        const data = JSON.parse(String(ev.data)) as {
          status?: string;
          error_message?: string;
        };
        if (data.status === "running" || data.status === "done" || data.status === "failed") {
          setState({
            status: data.status,
            message: data.error_message ?? null,
          });
        }
      } catch {
        /* ignore malformed payloads */
      }
    };

    return () => {
      if (ws) {
        ws.onmessage = null;
        ws.onerror = null;
        ws.onclose = null;
        if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
          ws.close();
        }
      }
    };
  }, [taskId]);

  return state;
}

export function useInferenceResult(inferenceId: string | undefined, enabled: boolean) {
  return useQuery<InferenceResultPublic>({
    queryKey: ["inference", inferenceId],
    queryFn: () =>
      InferenceService.getInferenceResultApiV1InferenceInferenceIdResultGet({
        inferenceId: inferenceId!,
      }),
    enabled: Boolean(inferenceId) && enabled,
  });
}

interface ImageState {
  image: HTMLImageElement | null;
  isLoading: boolean;
  error: Error | null;
}

export function useExplanationImage(
  inferenceId: string | undefined,
  enabled: boolean,
): ImageState {
  const [state, setState] = useState<ImageState>({
    image: null,
    isLoading: false,
    error: null,
  });

  useEffect(() => {
    if (!inferenceId || !enabled) {
      setState({ image: null, isLoading: false, error: null });
      return;
    }
    setState({ image: null, isLoading: true, error: null });
    const img = new Image();
    img.decoding = "async";
    let cancelled = false;
    img.onload = () => {
      if (!cancelled) setState({ image: img, isLoading: false, error: null });
    };
    img.onerror = () => {
      if (!cancelled) {
        setState({
          image: null,
          isLoading: false,
          error: new Error("Не удалось загрузить Grad-CAM"),
        });
      }
    };
    img.src = apiUrl(`/api/v1/inference/${inferenceId}/explanation`);
    return () => {
      cancelled = true;
      img.onload = null;
      img.onerror = null;
    };
  }, [inferenceId, enabled]);

  return state;
}
