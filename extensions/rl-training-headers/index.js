import { AsyncLocalStorage } from "node:async_hooks";

function resolveConfig(api) {
  const cfg = (api.pluginConfig ?? {});
  return {
    sessionIdHeader: cfg.sessionIdHeader ?? "X-Session-Id",
    turnTypeHeader: cfg.turnTypeHeader ?? "X-Turn-Type",
  };
}

const SIDE_TRIGGERS = new Set(["heartbeat", "memory", "cron"]);

export default function register(api) {
  const config = resolveConfig(api);
  const headerStore = new AsyncLocalStorage();

  const originalFetch = globalThis.fetch;

  globalThis.fetch = function rlPatchedFetch(input, init) {
    const merged = new Headers(init?.headers);
    merged.set("X-Pinggy-No-Screen", "true");
    
    const scopedHeaders = headerStore.getStore();
    if (scopedHeaders && init?.method?.toUpperCase() === "POST") {
      for (const [k, v] of Object.entries(scopedHeaders)) {
        if (!merged.has(k)) {
          merged.set(k, v);
        }
      }
    }
    return originalFetch.call(globalThis, input, { ...init, headers: merged });
  };

  api.on("before_prompt_build", (_event, ctx) => {
    const sessionId = ctx.sessionId ?? "";
    const turnType = SIDE_TRIGGERS.has(ctx.trigger ?? "") ? "side" : "main";
    headerStore.enterWith({
      [config.sessionIdHeader]: sessionId,
      [config.turnTypeHeader]: turnType,
    });
    return {};
  });

  api.logger.info("rl-training-headers: activated (fetch patched)");
}
