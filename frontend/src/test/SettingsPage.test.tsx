import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { SettingsPage } from "@/pages/SettingsPage";

const listModelsMock = vi.fn();
const getConfigMock = vi.fn();
const patchConfigMock = vi.fn();

vi.mock("@/api", async () => {
  const actual = await vi.importActual<typeof import("@/api")>("@/api");
  return {
    ...actual,
    ModelsService: {
      listModelsApiV1ModelsGet: (...args: unknown[]) => listModelsMock(...args),
    },
    ToolsService: {
      getToolConfigApiV1ToolsToolIdConfigGet: (args: unknown) => getConfigMock(args),
      patchToolConfigApiV1ToolsToolIdConfigPatch: (args: unknown) => patchConfigMock(args),
    },
  };
});

function renderPage() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={["/settings"]}>
        <Routes>
          <Route path="/settings" element={<SettingsPage />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

const PNEUMONIA = {
  tool_id: "pneumonia_classifier_v1",
  name: "RSNA Pneumonia Classifier",
  version: "1.0.0",
  description: "DenseNet-121",
  modality: "xray",
  task_type: "classification",
  input_shape: [3, 384, 384],
  class_names: ["Normal", "Pneumonia"],
  loaded: true,
};

const PNEUMONIA_CONFIG = {
  tool_id: "pneumonia_classifier_v1",
  schema: [
    {
      key: "mode",
      label: "Inference mode",
      type: "select" as const,
      default: "single",
      description: "Single = fast, ensemble = accurate.",
      options: [
        { value: "single", label: "Single model", description: "Fastest." },
        { value: "ensemble", label: "5-fold ensemble", description: "AUC 0.8927." },
      ],
    },
  ],
  values: { mode: "single" },
};

describe("SettingsPage", () => {
  beforeEach(() => {
    listModelsMock.mockReset();
    getConfigMock.mockReset();
    patchConfigMock.mockReset();
  });

  it("renders model card and the schema-driven mode select", async () => {
    listModelsMock.mockResolvedValueOnce([PNEUMONIA]);
    getConfigMock.mockResolvedValueOnce(PNEUMONIA_CONFIG);

    renderPage();

    await screen.findByText("RSNA Pneumonia Classifier");
    expect(await screen.findByText("Inference mode")).toBeInTheDocument();
    expect(screen.getByLabelText(/single model/i)).toBeChecked();
    expect(screen.getByLabelText(/5-fold ensemble/i)).not.toBeChecked();
  });

  it("save button is disabled until a value is changed", async () => {
    listModelsMock.mockResolvedValueOnce([PNEUMONIA]);
    getConfigMock.mockResolvedValueOnce(PNEUMONIA_CONFIG);

    renderPage();

    const save = await screen.findByRole("button", { name: /сохранить/i });
    expect(save).toBeDisabled();
  });

  it("submits only changed keys via PATCH", async () => {
    listModelsMock.mockResolvedValueOnce([PNEUMONIA]);
    getConfigMock.mockResolvedValueOnce(PNEUMONIA_CONFIG);
    patchConfigMock.mockResolvedValueOnce({
      ...PNEUMONIA_CONFIG,
      values: { mode: "ensemble" },
    });

    renderPage();

    const ensembleRadio = await screen.findByLabelText(/5-fold ensemble/i);
    await userEvent.click(ensembleRadio);

    const save = screen.getByRole("button", { name: /сохранить/i });
    expect(save).not.toBeDisabled();
    await userEvent.click(save);

    await waitFor(() => {
      expect(patchConfigMock).toHaveBeenCalledWith({
        toolId: "pneumonia_classifier_v1",
        requestBody: { values: { mode: "ensemble" } },
      });
    });
  });

  it("shows empty-schema notice for tools without settings", async () => {
    listModelsMock.mockResolvedValueOnce([
      { ...PNEUMONIA, tool_id: "no_settings", name: "NoSettings" },
    ]);
    getConfigMock.mockResolvedValueOnce({
      tool_id: "no_settings",
      schema: [],
      values: {},
    });

    renderPage();

    await screen.findByText(/не объявляет/i);
  });
});
