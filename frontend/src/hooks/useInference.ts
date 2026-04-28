import { useEffect, useRef, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
  InferenceService,
  type InferenceResultPublic,
  type InferenceSubmitResponse,
  type ModelListItem,
  ModelsService,
  TasksService,
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

const TERMINAL: TaskStatus[] = ["done", "failed"];
const POLL_INTERVAL_MS = 2000;

export function useTaskStatus(taskId: string | undefined): TaskState {
  const [state, setState] = useState<TaskState>({ status: "idle", message: null });
  // Keep a ref so polling callbacks can check current status without stale closure.
  const stateRef = useRef(state);
  stateRef.current = state;

  useEffect(() => {
    if (!taskId) {
      setState({ status: "idle", message: null });
      return;
    }
    setState({ status: "queued", message: null });

    let ws: WebSocket | null = null;
    let pollTimer: ReturnType<typeof setInterval> | null = null;

    const applyServerStatus = (status: string, message?: string | null) => {
      if (TERMINAL.includes(stateRef.current.status)) return;
      if (status === "running" || status === "done" || status === "failed") {
        setState({ status: status as TaskStatus, message: message ?? null });
      }
    };

    // Polling fallback — used when WS is unavailable or as a race-condition guard.
    const startPolling = () => {
      if (pollTimer !== null) return;
      pollTimer = setInterval(async () => {
        if (TERMINAL.includes(stateRef.current.status)) {
          if (pollTimer !== null) clearInterval(pollTimer);
          return;
        }
        try {
          const resp = await TasksService.getTaskStatusApiV1TasksTaskIdGet({ taskId: taskId! });
          applyServerStatus(resp.status);
          if (TERMINAL.includes(resp.status as TaskStatus)) {
            if (pollTimer !== null) clearInterval(pollTimer);
          }
        } catch {
          /* network blip — keep polling */
        }
      }, POLL_INTERVAL_MS);
    };

    try {
      ws = new WebSocket(wsUrl(`/ws/tasks/${taskId}`));
    } catch {
      setState({ status: "failed", message: "WebSocket недоступен" });
      startPolling();
      return;
    }

    ws.onmessage = (ev) => {
      try {
        const data = JSON.parse(String(ev.data)) as {
          status?: string;
          error_message?: string;
        };
        applyServerStatus(data.status ?? "", data.error_message);
      } catch {
        /* ignore malformed payloads */
      }
    };

    ws.onerror = () => {
      // WS failed — fall back to polling so the UI doesn't hang silently.
      startPolling();
    };

    ws.onclose = () => {
      // If the task isn't terminal yet (e.g. WS closed before task finished),
      // start polling to catch the final status.
      if (!TERMINAL.includes(stateRef.current.status)) {
        startPolling();
      }
    };

    return () => {
      if (pollTimer !== null) clearInterval(pollTimer);
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
  hasGradcam: boolean,
): ImageState {
  const [state, setState] = useState<ImageState>({
    image: null,
    isLoading: false,
    error: null,
  });

  useEffect(() => {
    if (!inferenceId || !enabled || !hasGradcam) {
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
  }, [inferenceId, enabled, hasGradcam]);

  return state;
}
