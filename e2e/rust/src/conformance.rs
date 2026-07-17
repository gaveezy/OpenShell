// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Driver-agnostic conformance scenarios for `OpenShell` gateway installations.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

use futures_util::StreamExt;
use miette::{IntoDiagnostic, Result, WrapErr};
use openshell_core::proto::{
    CreateSandboxRequest, DeleteSandboxRequest, ExecSandboxRequest, GetSandboxRequest,
    GpuResourceRequirements, ListSandboxesRequest, ResourceRequirements, SandboxPhase, SandboxSpec,
    WatchSandboxRequest, exec_sandbox_event, open_shell_client::OpenShellClient,
};
use openshell_core::{ObjectId, ObjectName};
use tonic::Code;
use tonic::transport::{Certificate, Channel, ClientTlsConfig, Endpoint, Identity};

type GrpcClient = OpenShellClient<Channel>;

/// Explicit connection material for an existing gateway installation.
#[derive(Clone, Debug, Default)]
pub struct ConnectionOptions {
    /// Path to the gateway CA certificate.
    pub tls_ca: Option<PathBuf>,
    /// Path to the client certificate.
    pub tls_cert: Option<PathBuf>,
    /// Path to the client private key.
    pub tls_key: Option<PathBuf>,
}

async fn grpc_client(server: &str, options: &ConnectionOptions) -> Result<GrpcClient> {
    let mut endpoint = Endpoint::from_shared(server.to_string())
        .into_diagnostic()?
        .connect_timeout(Duration::from_secs(10))
        .http2_adaptive_window(true)
        .http2_keep_alive_interval(Duration::from_secs(10))
        .keep_alive_while_idle(true);

    if server.starts_with("https://") {
        let ca_path = options
            .tls_ca
            .as_ref()
            .ok_or_else(|| miette::miette!("--tls-ca is required for HTTPS gateways"))?;
        let cert_path = options
            .tls_cert
            .as_ref()
            .ok_or_else(|| miette::miette!("--tls-cert is required for HTTPS gateways"))?;
        let key_path = options
            .tls_key
            .as_ref()
            .ok_or_else(|| miette::miette!("--tls-key is required for HTTPS gateways"))?;
        let ca = std::fs::read(ca_path)
            .into_diagnostic()
            .wrap_err_with(|| format!("failed to read TLS CA from {}", ca_path.display()))?;
        let cert = std::fs::read(cert_path)
            .into_diagnostic()
            .wrap_err_with(|| {
                format!(
                    "failed to read TLS certificate from {}",
                    cert_path.display()
                )
            })?;
        let key = std::fs::read(key_path)
            .into_diagnostic()
            .wrap_err_with(|| format!("failed to read TLS key from {}", key_path.display()))?;
        endpoint = endpoint
            .tls_config(
                ClientTlsConfig::new()
                    .ca_certificate(Certificate::from_pem(ca))
                    .identity(Identity::from_pem(cert, key)),
            )
            .into_diagnostic()?;
    }

    let channel = endpoint
        .connect()
        .await
        .into_diagnostic()
        .wrap_err("failed to connect to gateway")?;
    Ok(OpenShellClient::new(channel))
}

// -----------------------------------------------------------------------
// Conformance suite
// -----------------------------------------------------------------------

/// A single conformance scenario definition.
struct Scenario {
    /// Short lowercase-hyphenated name, used in sandbox naming and filtering.
    name: &'static str,
    /// Human-readable description shown in `conformance list`.
    description: &'static str,
}

fn all_scenarios() -> &'static [Scenario] {
    &[
        Scenario {
            name: "lifecycle",
            description: "Create → running → stop → delete completes without error",
        },
        Scenario {
            name: "not-found",
            description: "Get/stop/delete for an unknown sandbox ID returns an appropriate error",
        },
        Scenario {
            name: "idempotent-delete",
            description: "Deleting an already-deleted sandbox does not error",
        },
        Scenario {
            name: "validate",
            description: "Invalid sandbox specs are rejected before creation",
        },
        Scenario {
            name: "concurrent",
            description: "Two sandboxes created simultaneously do not interfere",
        },
        Scenario {
            name: "labels",
            description: "Labels are persisted on create and filter list results correctly",
        },
        Scenario {
            name: "exec",
            description: "A command executes in a ready sandbox and streams its output and exit status",
        },
        Scenario {
            name: "process-hardening",
            description: "Sandbox processes start with core dumps disabled",
        },
    ]
}

/// Outcome of a single conformance scenario.
#[derive(serde::Serialize)]
struct ScenarioResult {
    name: String,
    passed: bool,
    message: String,
    duration_ms: u64,
}

pub async fn conformance_run(
    server: &str,
    connection: &ConnectionOptions,
    filter: Option<&str>,
    timeout_secs: u64,
    output: &str,
) -> Result<()> {
    use std::time::Instant;

    let client = grpc_client(server, connection).await?;

    // Stable run-id for sandbox naming: seconds since Unix epoch, truncated
    // to 8 hex digits. Keeps names Kubernetes RFC 1123 safe and short enough
    // to read in `sandbox list` output.
    let run_id = format!(
        "{:08x}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
            & u64::from(u32::MAX)
    );

    let scenarios = all_scenarios()
        .iter()
        .filter(|s| filter.is_none_or(|f| s.name.contains(f)));

    let mut results: Vec<ScenarioResult> = Vec::new();
    let mut any_failed = false;

    for scenario in scenarios {
        let start = Instant::now();
        let timeout = Duration::from_secs(timeout_secs);
        let mut client = client.clone();

        let outcome =
            tokio::time::timeout(timeout, run_scenario(scenario.name, &mut client, &run_id))
                .await
                .unwrap_or_else(|_| {
                    Err(miette::miette!("scenario timed out after {timeout_secs}s"))
                });

        let passed = outcome.is_ok();
        if !passed {
            any_failed = true;
        }

        results.push(ScenarioResult {
            name: scenario.name.to_string(),
            passed,
            message: match &outcome {
                Ok(()) => "ok".to_string(),
                Err(e) => format!("{e}"),
            },
            duration_ms: u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX),
        });
    }

    match output {
        "json" => println!(
            "{}",
            serde_json::to_string_pretty(&results).into_diagnostic()?
        ),
        "yaml" => print!("{}", serde_yml::to_string(&results).into_diagnostic()?),
        "table" => {
            for result in &results {
                let status = if result.passed { "PASS" } else { "FAIL" };
                println!(
                    "  [{status}] {} ({}ms) — {}",
                    result.name, result.duration_ms, result.message
                );
            }
        }
        _ => return Err(miette::miette!("unsupported output format: {output}")),
    }

    if any_failed {
        return Err(miette::miette!("one or more conformance scenarios failed"));
    }
    Ok(())
}

/// Dispatch a scenario by name.
async fn run_scenario(name: &str, client: &mut GrpcClient, run_id: &str) -> Result<()> {
    match name {
        "lifecycle" => scenario_lifecycle(client, run_id).await,
        "not-found" => scenario_not_found(client, run_id).await,
        "idempotent-delete" => scenario_idempotent_delete(client, run_id).await,
        "validate" => scenario_validate(client).await,
        "concurrent" => scenario_concurrent(client, run_id).await,
        "labels" => scenario_labels(client, run_id).await,
        "exec" => scenario_exec(client, run_id).await,
        "process-hardening" => scenario_process_hardening(client, run_id).await,
        _ => Err(miette::miette!("scenario '{name}' is not yet implemented")),
    }
}

/// Poll `WatchSandbox` until the sandbox reaches Ready, returning an error on
/// Error phase or a closed stream. Does not perform cleanup — callers are
/// responsible for deleting the sandbox if this returns an error.
async fn wait_for_ready(
    client: &mut GrpcClient,
    sandbox_id: &str,
    sandbox_name: &str,
) -> Result<()> {
    let mut stream = client
        .watch_sandbox(WatchSandboxRequest {
            id: sandbox_id.to_string(),
            follow_status: true,
            follow_logs: false,
            follow_events: false,
            log_tail_lines: 0,
            event_tail: 0,
            stop_on_terminal: false,
            log_since_ms: 0,
            log_sources: vec![],
            log_min_level: String::new(),
        })
        .await
        .into_diagnostic()
        .wrap_err("watch_sandbox failed")?
        .into_inner();

    while let Some(item) = stream.next().await {
        let evt = item
            .into_diagnostic()
            .wrap_err("watch_sandbox stream error")?;
        if let Some(openshell_core::proto::sandbox_stream_event::Payload::Sandbox(s)) = evt.payload
        {
            match SandboxPhase::try_from(s.phase()).unwrap_or(SandboxPhase::Unknown) {
                SandboxPhase::Ready => return Ok(()),
                SandboxPhase::Error => {
                    return Err(miette::miette!(
                        "sandbox '{sandbox_name}' entered Error phase before becoming Ready"
                    ));
                }
                _ => {}
            }
        }
    }

    Err(miette::miette!(
        "watch stream ended before sandbox '{sandbox_name}' reached Ready"
    ))
}

/// Scenario: create → ready → delete.
///
/// Creates a minimal sandbox, waits for it to reach the Ready phase, then
/// deletes it. Verifies that the sandbox appears in the list between create
/// and delete, and that delete reports it as deleted.
async fn scenario_lifecycle(client: &mut GrpcClient, run_id: &str) -> Result<()> {
    let sandbox_name = format!("conformance-lifecycle-{run_id}");

    // ── 1. Create ────────────────────────────────────────────────────────
    let response = client
        .create_sandbox(CreateSandboxRequest {
            name: sandbox_name.clone(),
            spec: Some(SandboxSpec::default()),
            labels: HashMap::default(),
            annotations: HashMap::default(),
        })
        .await
        .into_diagnostic()
        .wrap_err("create_sandbox failed")?;

    let sandbox = response
        .into_inner()
        .sandbox
        .ok_or_else(|| miette::miette!("create_sandbox response missing sandbox"))?;
    let sandbox_id = sandbox.object_id().to_string();

    // ── 2. Wait for Ready ────────────────────────────────────────────────
    if let Err(e) = wait_for_ready(client, &sandbox_id, &sandbox_name).await {
        let _ = client
            .delete_sandbox(DeleteSandboxRequest {
                name: sandbox_name.clone(),
            })
            .await;
        return Err(e);
    }

    // ── 3. Verify it appears in the list ────────────────────────────────
    let list_response = client
        .list_sandboxes(ListSandboxesRequest::default())
        .await
        .into_diagnostic()
        .wrap_err("list_sandboxes failed")?;

    let found = list_response
        .into_inner()
        .sandboxes
        .iter()
        .any(|s| s.object_name() == sandbox_name);

    if !found {
        let _ = client
            .delete_sandbox(DeleteSandboxRequest {
                name: sandbox_name.clone(),
            })
            .await;
        return Err(miette::miette!(
            "sandbox '{sandbox_name}' not found in list_sandboxes response after creation"
        ));
    }

    // ── 4. Delete ────────────────────────────────────────────────────────
    let del_response = client
        .delete_sandbox(DeleteSandboxRequest {
            name: sandbox_name.clone(),
        })
        .await
        .into_diagnostic()
        .wrap_err("delete_sandbox failed")?;

    if !del_response.into_inner().deleted {
        return Err(miette::miette!(
            "delete_sandbox reported sandbox '{sandbox_name}' was not deleted"
        ));
    }

    Ok(())
}

/// Scenario: get and delete a sandbox that does not exist.
///
/// Verifies that `GetSandbox` returns `NOT_FOUND` and that `DeleteSandbox`
/// returns `deleted: false` without erroring for a name that was never
/// created.
async fn scenario_not_found(client: &mut GrpcClient, run_id: &str) -> Result<()> {
    let phantom_name = format!("conformance-not-found-{run_id}");

    // ── 1. GetSandbox → NOT_FOUND ────────────────────────────────────────
    let err = client
        .get_sandbox(GetSandboxRequest {
            name: phantom_name.clone(),
        })
        .await
        .expect_err("get_sandbox on a non-existent sandbox should have returned NOT_FOUND");

    if err.code() != Code::NotFound {
        return Err(miette::miette!(
            "get_sandbox returned {} instead of NOT_FOUND",
            err.code()
        ));
    }

    // ── 2. DeleteSandbox → ok, deleted: false ────────────────────────────
    let del = client
        .delete_sandbox(DeleteSandboxRequest {
            name: phantom_name.clone(),
        })
        .await
        .into_diagnostic()
        .wrap_err("delete_sandbox on a non-existent sandbox should not error")?
        .into_inner();

    if del.deleted {
        return Err(miette::miette!(
            "delete_sandbox reported deleted=true for a sandbox that was never created"
        ));
    }

    Ok(())
}

/// Scenario: delete an already-deleted sandbox does not error.
///
/// Creates a sandbox, waits for it to be Ready, deletes it (expecting
/// `deleted: true`), then deletes it again and asserts the second call
/// returns `deleted: false` without an error.
async fn scenario_idempotent_delete(client: &mut GrpcClient, run_id: &str) -> Result<()> {
    let sandbox_name = format!("conformance-idempotent-delete-{run_id}");

    // ── 1. Create ────────────────────────────────────────────────────────
    let response = client
        .create_sandbox(CreateSandboxRequest {
            name: sandbox_name.clone(),
            spec: Some(SandboxSpec::default()),
            labels: HashMap::default(),
            annotations: HashMap::default(),
        })
        .await
        .into_diagnostic()
        .wrap_err("create_sandbox failed")?;

    let sandbox = response
        .into_inner()
        .sandbox
        .ok_or_else(|| miette::miette!("create_sandbox response missing sandbox"))?;
    let sandbox_id = sandbox.object_id().to_string();

    // ── 2. Wait for Ready ────────────────────────────────────────────────
    if let Err(e) = wait_for_ready(client, &sandbox_id, &sandbox_name).await {
        let _ = client
            .delete_sandbox(DeleteSandboxRequest {
                name: sandbox_name.clone(),
            })
            .await;
        return Err(e);
    }

    // ── 3. First delete → deleted: true ──────────────────────────────────
    let del1 = client
        .delete_sandbox(DeleteSandboxRequest {
            name: sandbox_name.clone(),
        })
        .await
        .into_diagnostic()
        .wrap_err("first delete_sandbox failed")?
        .into_inner();

    if !del1.deleted {
        return Err(miette::miette!(
            "first delete_sandbox reported deleted=false for sandbox '{sandbox_name}'"
        ));
    }

    // ── 4. Second delete → ok, deleted: false ────────────────────────────
    let del2 = client
        .delete_sandbox(DeleteSandboxRequest {
            name: sandbox_name.clone(),
        })
        .await
        .into_diagnostic()
        .wrap_err("second delete_sandbox (idempotency check) returned an error")?
        .into_inner();

    if del2.deleted {
        return Err(miette::miette!(
            "second delete_sandbox reported deleted=true — expected deleted=false"
        ));
    }

    Ok(())
}

/// Scenario: invalid sandbox specs are rejected before creation.
///
/// Verifies that the gateway returns `INVALID_ARGUMENT` for two distinct
/// invalid inputs without creating any sandbox: a missing spec field and a
/// zero GPU count.
async fn scenario_validate(client: &mut GrpcClient) -> Result<()> {
    // ── 1. spec=None → INVALID_ARGUMENT ──────────────────────────────────
    let err = client
        .create_sandbox(CreateSandboxRequest {
            name: String::new(),
            spec: None,
            labels: HashMap::default(),
            annotations: HashMap::default(),
        })
        .await
        .expect_err("create_sandbox with spec=None should have been rejected");

    if err.code() != Code::InvalidArgument {
        return Err(miette::miette!(
            "create_sandbox(spec=None) returned {} instead of INVALID_ARGUMENT",
            err.code()
        ));
    }

    // ── 2. gpu.count=0 → INVALID_ARGUMENT ────────────────────────────────
    let err2 = client
        .create_sandbox(CreateSandboxRequest {
            name: String::new(),
            spec: Some(SandboxSpec {
                resource_requirements: Some(ResourceRequirements {
                    gpu: Some(GpuResourceRequirements { count: Some(0) }),
                }),
                ..Default::default()
            }),
            labels: HashMap::default(),
            annotations: HashMap::default(),
        })
        .await
        .expect_err("create_sandbox with gpu.count=0 should have been rejected");

    if err2.code() != Code::InvalidArgument {
        return Err(miette::miette!(
            "create_sandbox(gpu.count=0) returned {} instead of INVALID_ARGUMENT",
            err2.code()
        ));
    }

    Ok(())
}

/// Scenario: two sandboxes created simultaneously do not interfere.
///
/// Issues two `CreateSandbox` calls concurrently, waits for both to reach
/// Ready in parallel, verifies both appear in `ListSandboxes`, then deletes
/// both. Any failure cleans up both sandboxes before returning.
async fn scenario_concurrent(client: &mut GrpcClient, run_id: &str) -> Result<()> {
    let name_a = format!("conformance-concurrent-a-{run_id}");
    let name_b = format!("conformance-concurrent-b-{run_id}");

    // ── 1. Create both sandboxes concurrently ────────────────────────────
    let mut client_a = client.clone();
    let mut client_b = client.clone();

    let (resp_a, resp_b) = tokio::join!(
        client_a.create_sandbox(CreateSandboxRequest {
            name: name_a.clone(),
            spec: Some(SandboxSpec::default()),
            labels: HashMap::default(),
            annotations: HashMap::default(),
        }),
        client_b.create_sandbox(CreateSandboxRequest {
            name: name_b.clone(),
            spec: Some(SandboxSpec::default()),
            labels: HashMap::default(),
            annotations: HashMap::default(),
        }),
    );

    let sandbox_a = resp_a
        .into_diagnostic()
        .wrap_err("create_sandbox(a) failed")?
        .into_inner()
        .sandbox
        .ok_or_else(|| miette::miette!("create_sandbox(a) response missing sandbox"))?;
    let sandbox_b = resp_b
        .into_diagnostic()
        .wrap_err("create_sandbox(b) failed")?
        .into_inner()
        .sandbox
        .ok_or_else(|| miette::miette!("create_sandbox(b) response missing sandbox"))?;

    let id_a = sandbox_a.object_id().to_string();
    let id_b = sandbox_b.object_id().to_string();

    // ── 2. Wait for both to reach Ready concurrently ─────────────────────
    let (ready_a, ready_b) = tokio::join!(
        wait_for_ready(&mut client_a, &id_a, &name_a),
        wait_for_ready(&mut client_b, &id_b, &name_b),
    );

    // Best-effort cleanup before surfacing watch errors.
    if ready_a.is_err() || ready_b.is_err() {
        let _ = client
            .delete_sandbox(DeleteSandboxRequest {
                name: name_a.clone(),
            })
            .await;
        let _ = client
            .delete_sandbox(DeleteSandboxRequest {
                name: name_b.clone(),
            })
            .await;
        ready_a?;
        ready_b?;
    }

    // ── 3. Both appear in the list ───────────────────────────────────────
    let sandboxes = client
        .list_sandboxes(ListSandboxesRequest::default())
        .await
        .into_diagnostic()
        .wrap_err("list_sandboxes failed")?
        .into_inner()
        .sandboxes;

    let found_a = sandboxes.iter().any(|s| s.object_name() == name_a);
    let found_b = sandboxes.iter().any(|s| s.object_name() == name_b);

    // ── 4. Delete both ───────────────────────────────────────────────────
    let _ = client
        .delete_sandbox(DeleteSandboxRequest {
            name: name_a.clone(),
        })
        .await;
    let _ = client
        .delete_sandbox(DeleteSandboxRequest {
            name: name_b.clone(),
        })
        .await;

    if !found_a {
        return Err(miette::miette!(
            "sandbox '{name_a}' not found in list_sandboxes after concurrent create"
        ));
    }
    if !found_b {
        return Err(miette::miette!(
            "sandbox '{name_b}' not found in list_sandboxes after concurrent create"
        ));
    }

    Ok(())
}

/// Scenario: labels are persisted on create and filter list results correctly.
///
/// Creates two sandboxes with distinct label values, calls `ListSandboxes`
/// with a label selector, and verifies that only the matching sandbox is
/// returned. Labels are stored by the gateway on creation so no Ready wait
/// is required; both sandboxes are deleted after the assertion.
async fn scenario_labels(client: &mut GrpcClient, run_id: &str) -> Result<()> {
    let name_a = format!("conformance-labels-a-{run_id}");
    let name_b = format!("conformance-labels-b-{run_id}");
    let label_key = "conformance-scenario".to_string();

    // ── 1. Create two sandboxes with distinct label values ───────────────
    client
        .create_sandbox(CreateSandboxRequest {
            name: name_a.clone(),
            spec: Some(SandboxSpec::default()),
            labels: [(label_key.clone(), "labels-a".to_string())]
                .into_iter()
                .collect(),
            annotations: HashMap::default(),
        })
        .await
        .into_diagnostic()
        .wrap_err("create_sandbox(a) failed")?;

    client
        .create_sandbox(CreateSandboxRequest {
            name: name_b.clone(),
            spec: Some(SandboxSpec::default()),
            labels: [(label_key.clone(), "labels-b".to_string())]
                .into_iter()
                .collect(),
            annotations: HashMap::default(),
        })
        .await
        .into_diagnostic()
        .wrap_err("create_sandbox(b) failed")?;

    // ── 2. Filter by label — must return only sandbox A ──────────────────
    let filtered = client
        .list_sandboxes(ListSandboxesRequest {
            label_selector: format!("{label_key}=labels-a"),
            ..Default::default()
        })
        .await
        .into_diagnostic()
        .wrap_err("list_sandboxes with label_selector failed")?
        .into_inner()
        .sandboxes;

    // ── 3. Cleanup ───────────────────────────────────────────────────────
    let _ = client
        .delete_sandbox(DeleteSandboxRequest {
            name: name_a.clone(),
        })
        .await;
    let _ = client
        .delete_sandbox(DeleteSandboxRequest {
            name: name_b.clone(),
        })
        .await;

    // ── 4. Assert after cleanup so both sandboxes are always removed ──────
    let found_a = filtered.iter().any(|s| s.object_name() == name_a);
    let found_b = filtered.iter().any(|s| s.object_name() == name_b);

    if !found_a {
        return Err(miette::miette!(
            "sandbox '{name_a}' not found in label-filtered list (selector: {label_key}=labels-a)"
        ));
    }
    if found_b {
        return Err(miette::miette!(
            "sandbox '{name_b}' appeared in label-filtered list but should have been excluded \
             (selector: {label_key}=labels-a)"
        ));
    }

    Ok(())
}

struct ExecResult {
    stdout: String,
    stderr: String,
    exit_code: Option<i32>,
}

async fn exec_command(
    client: &mut GrpcClient,
    sandbox_id: &str,
    command: Vec<String>,
) -> Result<ExecResult> {
    let mut stream = client
        .exec_sandbox(ExecSandboxRequest {
            sandbox_id: sandbox_id.to_string(),
            command,
            ..Default::default()
        })
        .await
        .into_diagnostic()
        .wrap_err("exec_sandbox failed")?
        .into_inner();

    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut exit_code = None;

    while let Some(event) = stream.next().await {
        match event
            .into_diagnostic()
            .wrap_err("exec_sandbox stream error")?
            .payload
        {
            Some(exec_sandbox_event::Payload::Stdout(chunk)) => stdout.extend(chunk.data),
            Some(exec_sandbox_event::Payload::Stderr(chunk)) => stderr.extend(chunk.data),
            Some(exec_sandbox_event::Payload::Exit(exit)) => exit_code = Some(exit.exit_code),
            None => {}
        }
    }

    Ok(ExecResult {
        stdout: String::from_utf8_lossy(&stdout).into_owned(),
        stderr: String::from_utf8_lossy(&stderr).into_owned(),
        exit_code,
    })
}

async fn create_ready_sandbox(client: &mut GrpcClient, sandbox_name: &str) -> Result<String> {
    let response = client
        .create_sandbox(CreateSandboxRequest {
            name: sandbox_name.to_string(),
            spec: Some(SandboxSpec::default()),
            labels: HashMap::default(),
            annotations: HashMap::default(),
        })
        .await
        .into_diagnostic()
        .wrap_err("create_sandbox failed")?;

    let sandbox = response
        .into_inner()
        .sandbox
        .ok_or_else(|| miette::miette!("create_sandbox response missing sandbox"))?;
    let sandbox_id = sandbox.object_id().to_string();

    if let Err(error) = wait_for_ready(client, &sandbox_id, sandbox_name).await {
        let _ = client
            .delete_sandbox(DeleteSandboxRequest {
                name: sandbox_name.to_string(),
            })
            .await;
        return Err(error);
    }

    Ok(sandbox_id)
}

/// Scenario: execute a command in a ready sandbox through the gateway API.
///
/// Verifies stdout and the final exit status independently, then deletes the
/// sandbox regardless of whether command execution succeeds.
async fn scenario_exec(client: &mut GrpcClient, run_id: &str) -> Result<()> {
    const OUTPUT_MARKER: &str = "conformance-exec-ok";

    let sandbox_name = format!("conformance-exec-{run_id}");
    let sandbox_id = create_ready_sandbox(client, &sandbox_name).await?;

    let exec_result = async {
        let result = exec_command(
            client,
            &sandbox_id,
            vec![
                "sh".to_string(),
                "-c".to_string(),
                format!("printf {OUTPUT_MARKER}"),
            ],
        )
        .await?;
        if !result.stdout.contains(OUTPUT_MARKER) {
            return Err(miette::miette!(
                "exec_sandbox stdout did not contain '{OUTPUT_MARKER}'; stdout={:?}, stderr={:?}",
                result.stdout,
                result.stderr,
            ));
        }
        if result.exit_code != Some(0) {
            return Err(miette::miette!(
                "exec_sandbox returned exit code {:?}; stdout={:?}, stderr={:?}",
                result.exit_code,
                result.stdout,
                result.stderr,
            ));
        }

        Ok(())
    }
    .await;

    let cleanup_result = client
        .delete_sandbox(DeleteSandboxRequest { name: sandbox_name })
        .await
        .into_diagnostic()
        .wrap_err("delete_sandbox after exec failed");

    exec_result?;
    cleanup_result?;
    Ok(())
}

/// Scenario: sandbox processes start with core dumps disabled.
async fn scenario_process_hardening(client: &mut GrpcClient, run_id: &str) -> Result<()> {
    const OUTPUT_MARKER: &str = "core-limit-ok";

    let sandbox_name = format!("conformance-process-hardening-{run_id}");
    let sandbox_id = create_ready_sandbox(client, &sandbox_name).await?;
    let command_result = exec_command(
        client,
        &sandbox_id,
        vec![
            "sh".to_string(),
            "-lc".to_string(),
            format!("test \"$(ulimit -c)\" = 0 && printf {OUTPUT_MARKER}"),
        ],
    )
    .await;

    let cleanup_result = client
        .delete_sandbox(DeleteSandboxRequest { name: sandbox_name })
        .await
        .into_diagnostic()
        .wrap_err("delete_sandbox after process hardening check failed");

    let result = command_result?;
    cleanup_result?;
    if result.exit_code != Some(0) || !result.stdout.contains(OUTPUT_MARKER) {
        return Err(miette::miette!(
            "sandbox process core-dump check failed with exit code {:?}; stdout={:?}, stderr={:?}",
            result.exit_code,
            result.stdout,
            result.stderr,
        ));
    }

    Ok(())
}

pub fn conformance_list(output: &str) -> Result<()> {
    let scenarios = all_scenarios();

    match output {
        "json" => {
            let values: Vec<_> = scenarios
                .iter()
                .map(|scenario| {
                    serde_json::json!({
                        "name": scenario.name,
                        "description": scenario.description,
                    })
                })
                .collect();
            println!(
                "{}",
                serde_json::to_string_pretty(&values).into_diagnostic()?
            );
        }
        "yaml" => {
            let values: Vec<_> = scenarios
                .iter()
                .map(|scenario| {
                    serde_json::json!({
                        "name": scenario.name,
                        "description": scenario.description,
                    })
                })
                .collect();
            print!("{}", serde_yml::to_string(&values).into_diagnostic()?);
        }
        "table" => {
            println!("{} conformance scenarios:", scenarios.len());
            for scenario in scenarios {
                println!("  {:<20} {}", scenario.name, scenario.description);
            }
        }
        _ => return Err(miette::miette!("unsupported output format: {output}")),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::all_scenarios;

    #[test]
    fn ci_scenario_set_contains_only_implemented_scenarios() {
        let names = all_scenarios()
            .iter()
            .map(|scenario| scenario.name)
            .collect::<Vec<_>>();

        assert_eq!(
            names,
            [
                "lifecycle",
                "not-found",
                "idempotent-delete",
                "validate",
                "concurrent",
                "labels",
                "exec",
                "process-hardening",
            ]
        );
    }
}
