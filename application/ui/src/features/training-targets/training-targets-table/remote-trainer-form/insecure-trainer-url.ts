// A Hugging Face token is sent to a remote trainer's URL on every job
// submission. Plain http:// leaves it readable in transit to anything but a
// loopback trainer, so warn rather than block the (still supported)
// loopback/private-network http:// deployment outright.
export const isInsecureTrainerUrl = (url: string): boolean =>
    /^http:\/\/(?!localhost|127\.0\.0\.1|\[::1\])/i.test(url.trim());

export const INSECURE_TRAINER_URL_WARNING =
    'Warning: http:// sends your Hugging Face token to this trainer unencrypted. ' +
    'Use https:// unless the trainer is only reachable on localhost/a trusted private network.';
