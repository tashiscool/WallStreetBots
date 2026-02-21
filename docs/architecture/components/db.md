# Component: db
## Layer: infrastructure
## Responsibilities: Owns db concerns and exposes stable behavior to upstream callers.
## Interfaces:
- Inputs: request payloads, configs, scheduled triggers
- Outputs: domain responses, side-effects, persisted state
- Events: emits operational and domain events where applicable
## Dependencies:
- Internal: tbd
- External: tbd
## Constraints:
- Preserve backward compatibility unless ADR-approved.
- Maintain explicit validation and error handling.
- Keep security-sensitive operations auditable.
## Not Responsible For:
- Cross-domain orchestration owned by other components.
- Data contracts outside this component boundary.
## Files:
- scripts/db_setup.sql
- db/taxlots.sql
- staticfiles/admin/js/SelectFilter2.bdb8d0cc579e.js
- staticfiles/admin/js/vendor/select2/i18n/ne.3d79fd3f08db.js
- staticfiles/admin/js/vendor/select2/i18n/es.66dbc2652fb1.js
- staticfiles/assets/js/core/popper.min.50dbd6ab5b2f.js
- staticfiles/assets/js/plugins/perfect-scrollbar.min.7726dbb47415.js
## Arch Critical Files:
- tbd
